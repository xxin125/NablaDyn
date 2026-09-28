"""The public constructors of the potentials; the classes they build stay private."""

from collections.abc import Mapping
from numbers import Real

import torch
from torch import Tensor

from ._base import Potential
from ._lennard_jones import LennardJones
from ._opls import OPLS, Parameters, Topology
from ._schnet import SchNet
from ._tabulated import Tabulated


def lennard_jones(
    *,
    atom_types: Tensor,
    sigma: Tensor,
    epsilon: Tensor,
    cutoff: Real,
    trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
) -> Potential:
    """Build a 12-6 Lennard-Jones model shifted to zero at the cutoff.

    E sums u(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] - u_c over the pairs closer than the
    cutoff rc, with u_c = 4 eps [(sigma/rc)^12 - (sigma/rc)^6] of each pair's own sigma and
    epsilon. E is continuous at rc and is the energy MD conserves and samples. u_c is
    constant, so forces and virial are those of the unshifted potential; the force steps
    to zero at rc.

    ``atom_types`` [N], int64 in [0, T), fixes the atoms. ``sigma`` (nm) and ``epsilon``
    (kJ/mol) hold T(T+1)/2 values each, one per unordered type pair in the order (0, 0),
    (0, 1), ..., (1, 1), ...; no mixing rule is applied. ``cutoff`` is in nm. ``trainable``
    names what a fit trains, among "sigma" and "epsilon": a tuple trains every row of each;
    a mapping trains only the listed rows, as {"epsilon": [1]} for (0, 1), and holds the
    rest. Only named parameters require gradients. The model keeps copies of its inputs and
    computes in the dtype and on the device of ``sigma``. Syncs with the host.

    Raises ``ValueError`` for a ``trainable`` that names other parameters or rows outside
    [0, T(T+1)/2), a cutoff that is not finite and positive, parameters that are not finite
    positive float vectors of T(T+1)/2 values, or atom types outside [0, T).
    """
    return LennardJones(
        atom_types=atom_types,
        sigma=sigma,
        epsilon=epsilon,
        cutoff=cutoff,
        trainable=trainable,
    )


def opls(
    *,
    topology: Topology,
    parameters: Parameters,
    cutoff: Real,
    cell: Tensor,
    ewald_tolerance: Real = 5e-4,
) -> Potential:
    """Build an OPLS-AA model of a fixed topology, with Coulomb by smooth particle-mesh Ewald.

    U = sum_bonds 0.5 k (r - r0)^2 + sum_angles 0.5 k (theta - theta0)^2 +
    sum_dihedrals k [1 + cos(n phi - phase)] + U_LJ + U_C, each term with its type's row of
    ``parameters``. theta = atan2(|a x b|, a . b) with a = r_i - r_j and b = r_k - r_j; phi
    is the IUPAC dihedral of i-j-k-l in (-pi, pi], 0 cis and pi trans. Every vector is the
    minimum image, so a molecule must span less than half of each cell length.

    ``topology.bonds`` sets the pair classes. Pairs 1 or 2 bonds apart do not interact; 1-4
    pairs, 3 bonds apart, count 0.5 u(r) + 0.5 C q_i q_j / r at any distance, unshifted.
    Other pairs within the cutoff count u(r) - u(rc) and the real-space Ewald term
    C q_i q_j [erfc(alpha r)/r - erfc(alpha rc)/rc], both zero at rc; an order-5 smooth PME
    mesh adds every pair's long-range part, less the self and net-charge terms. Here
    u(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] with sigma = sqrt(sigma_a sigma_b) and
    eps = sqrt(eps_a eps_b) of the two atoms' LJ types; a zero product gives no term, and a
    zero sigma or epsilon gets zero gradient. C is ``units.COULOMB``; q are the charges.

    ``cell`` [3], in nm, plans the Ewald splitting once for ``ewald_tolerance`` delta, about
    the relative force error: alpha = sqrt(-ln(2 delta)) / cutoff, and about
    2 alpha L / (3 delta^(1/5)) mesh points per axis, rounded up to a product of 2, 3 and 5.
    Both stay fixed as ``model.alpha`` (nm^-1) and ``model.mesh``, so accuracy drops silently
    in a larger cell: plan with the largest cell a run visits.

    ``topology`` is an ``opls.Topology`` and ``parameters`` an ``opls.Parameters``; their
    docstrings give fields, shapes and units. ``cutoff`` (nm) holds for both the LJ and the
    real-space terms. The model keeps copies of the tables as frozen parameters and of the
    topology as buffers, and computes in the dtype and on the device of ``parameters``.
    Syncs with the host.

    Raises ``ValueError`` for a ``topology`` or ``parameters`` of another type, a cutoff that
    is not finite and positive, a cell that is not three finite positive lengths, a tolerance
    outside (0, 0.5), tables that are not finite vectors of one float dtype and device,
    paired tables of different lengths, topology fields of another dtype or shape, atoms or
    types out of range, sigma or epsilon below 0, or periodicities below 1.
    """
    return OPLS(
        topology=topology,
        parameters=parameters,
        cutoff=cutoff,
        cell=cell,
        ewald_tolerance=ewald_tolerance,
    )


opls.Topology = Topology
opls.Parameters = Parameters


def schnet(
    *,
    atom_types: Tensor,
    num_types: int,
    cutoff: Real,
    features: int = 64,
    interactions: int = 3,
    gaussians: int = 50,
    trainable: bool = False,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Potential:
    """Build a SchNet network (Schütt et al., J. Chem. Phys. 148, 241722, 2018).

    Each atom starts from its type's row of a table [T, F], x_i = table[t_i]. Each of the
    ``interactions`` blocks adds ssp(m_i D1 + d1) D2 + d2, where m_i sums f_ij x_j A over
    the pairs within the cutoff, each pair sending to both its atoms. The filter is
    f_ij = (ssp(g(r_ij) B1 + b1) B2 + b2) C(r_ij), with K = ``gaussians`` fixed Gaussians
    g_k(r) = exp(-(r - k D)^2 / (2 D^2)), D = cutoff / (K - 1), and the envelope
    C(r) = (1 + cos(pi r / cutoff)) / 2, so a pair's contribution to the energy and forces
    vanishes smoothly at the cutoff. The energy is E = sum_i ssp(x_i R1 + e1) R2 + e2 in
    kJ/mol, with ssp(x) = ln(1 + e^x) - ln 2.

    ``atom_types`` [N], int64 in [0, ``num_types``), fixes the atoms and the device;
    ``cutoff`` is in nm. ``features`` F, even, sizes the table, filters and blocks; the
    readout's hidden layer is F / 2. ``trainable`` gives every weight a gradient, or none.
    Weights are drawn in float64 from ``seed`` (table standard normal, matrices
    Xavier-uniform, biases 0) and cast to ``dtype``, so one seed gives one network in both
    dtypes. The state dict holds only the weights, so fitted weights load into a network of
    other atoms with the same types and sizes. Syncs with the host.

    Raises ``ValueError`` for sizes that are not integers in range or an odd ``features``,
    a cutoff that is not finite and positive, a ``dtype`` other than float32 or float64,
    or atom types outside [0, ``num_types``).
    """
    return SchNet(
        atom_types=atom_types,
        num_types=num_types,
        cutoff=cutoff,
        features=features,
        interactions=interactions,
        gaussians=gaussians,
        trainable=trainable,
        seed=seed,
        dtype=dtype,
    )


def tabulated(
    *,
    atom_types: Tensor,
    table: Tensor,
    start: Tensor,
    cutoff: Real,
    trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
) -> Potential:
    """Build a pair potential tabulated per unordered type pair, 0 at the cutoff.

    E sums u(r) over the pairs closer than the cutoff, each reading the row of its two
    types. Between knots u is a C1 Catmull-Rom cubic, exact for any quadratic; below a
    row's first knot u follows its tangent. Rows end at 0 at the cutoff, so the energy is
    continuous there; the force steps to zero.

    ``atom_types`` [N], int64 in [0, T), fixes the atoms. ``table`` [P, K], P = T(T+1)/2 and
    K >= 4, holds u in kJ/mol at each row's knots, one row per unordered type pair in the
    order (0, 0), (0, 1), ..., (1, 1), ...; its last column, at the cutoff, must be 0. Row
    p's knots lie at start_p + k h_p, h_p = (cutoff - start_p) / (K - 1), with ``start`` [P]
    and ``cutoff`` in nm. ``trainable`` is ("table",) to train every row or {"table": rows}
    to train only those; no fit moves the last knot. The model computes in the dtype and on
    the device of ``table``. Syncs with the host.

    Raises ``ValueError`` for a ``trainable`` that names another parameter or rows outside
    [0, P), a cutoff that is not finite and positive, a table that is not a finite float
    matrix of T(T+1)/2 rows and at least 4 columns ending in 0, a ``start`` that is not [P]
    in the table's dtype and device or lies outside [0, cutoff), or atom types outside
    [0, T).
    """
    return Tabulated(
        atom_types=atom_types,
        table=table,
        start=start,
        cutoff=cutoff,
        trainable=trainable,
    )


__all__ = ("lennard_jones", "opls", "schnet", "tabulated")
