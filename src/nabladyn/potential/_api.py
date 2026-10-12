"""The public constructors of the potentials; the classes they build stay private."""

from collections.abc import Mapping
from numbers import Real

import torch
from torch import Tensor

from ._angular_schnet import AngularSchNet
from ._base import Potential
from ._lennard_jones import LennardJones
from ._opls import OPLS, Parameters, Topology
from ._schnet import SchNet
from ._sum import Sum
from ._tabulated import Tabulated


def lennard_jones(
    *,
    atom_types: Tensor,
    sigma: Tensor,
    epsilon: Tensor,
    cutoff: Real,
    trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
) -> Potential:
    r"""Build a 12-6 Lennard-Jones model shifted to zero at the cutoff.

    E sums, over the pairs closer than the cutoff $`r_c`$,

    ```math
    u(r) = 4\varepsilon [(\sigma/r)^{12} - (\sigma/r)^6] - u_c,
    \quad u_c = 4\varepsilon [(\sigma/r_c)^{12} - (\sigma/r_c)^6],
    ```

    with each pair's own sigma and epsilon. E is continuous at $`r_c`$ and is the energy MD
    conserves and samples. $`u_c`$ is constant, so forces and virial are those of the
    unshifted potential; the force steps to zero at $`r_c`$.

    ``atom_types`` [N], int64 in [0, T), fixes the atoms. ``sigma`` (nm) and ``epsilon``
    (kJ/mol) hold $`T(T+1)/2`$ values each, one per unordered type pair in the order (0, 0),
    (0, 1), ..., (1, 1), ...; no mixing rule is applied. ``cutoff`` is in nm. ``trainable``
    names what a fit trains, among "sigma" and "epsilon": a tuple trains every row of each;
    a mapping trains only the listed rows, as {"epsilon": [1]} for (0, 1), and holds the
    rest. Only named parameters require gradients. The model keeps copies of its inputs and
    computes in the dtype and on the device of ``sigma``. Syncs with the host.

    Raises ``ValueError`` for a ``trainable`` that names other parameters or rows outside
    [0, T(T+1)/2), a cutoff that is not finite and positive, parameters that are not finite
    positive float vectors of $`T(T+1)/2`$ values, or atom types outside [0, T).
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
    r"""Build an OPLS-AA model of a fixed topology, with Coulomb by smooth particle-mesh Ewald.

    ```math
    U = \sum_{\mathrm{bonds}} 0.5 k (r - r_0)^2
    + \sum_{\mathrm{angles}} 0.5 k (\theta - \theta_0)^2
    + \sum_{\mathrm{dihedrals}} k [1 + \cos(n\phi - \mathrm{phase})]
    + U_{\mathrm{LJ}} + U_{\mathrm{C}},
    ```

    each term with its type's row of ``parameters``. The angle is

    ```math
    \theta = \mathrm{atan2}(\lvert a \times b \rvert, a \cdot b)
    ```

    with $`a = r_i - r_j`$ and $`b = r_k - r_j`$; $`\phi`$ is the IUPAC dihedral of i-j-k-l in
    $`(-\pi, \pi]`$, 0 cis and $`\pi`$ trans. Every vector is the minimum image, so a molecule
    must span less than half of each cell length.

    ``topology.bonds`` sets the pair classes. Pairs 1 or 2 bonds apart do not interact; 1-4
    pairs, 3 bonds apart, count $`0.5 u(r) + 0.5 C q_i q_j / r`$ at any distance, unshifted.
    Other pairs within the cutoff $`r_c`$ count $`u(r) - u(r_c)`$ and the real-space Ewald term

    ```math
    C q_i q_j [\mathrm{erfc}(\alpha r)/r - \mathrm{erfc}(\alpha r_c)/r_c],
    ```

    both zero at $`r_c`$; an order-5 smooth PME mesh adds every pair's long-range part, less
    the self and net-charge terms. Here $`u(r) = 4\varepsilon [(\sigma/r)^{12} - (\sigma/r)^6]`$
    with $`\sigma = \sqrt{\sigma_a \sigma_b}`$ and
    $`\varepsilon = \sqrt{\varepsilon_a \varepsilon_b}`$ of the two atoms' LJ types; a zero
    product gives no term, and a zero sigma or epsilon gets zero gradient. C is
    ``units.COULOMB``; q are the charges.

    ``cell`` [3], in nm, plans the Ewald splitting once for ``ewald_tolerance`` $`\delta`$,
    about the relative force error: $`\alpha = \sqrt{-\ln(2\delta)} / r_c`$, and about
    $`2\alpha L / (3\delta^{1/5})`$ mesh points per axis, rounded up to a product of 2, 3 and
    5. Both stay fixed as ``model.alpha`` (nm^-1) and ``model.mesh``, so accuracy drops silently
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
    r"""Build a SchNet network (Schütt et al., J. Chem. Phys. 148, 241722, 2018).

    Each atom starts from its type's row of a table [T, F], $`x_i = \mathrm{table}[t_i]`$.
    Each of the ``interactions`` blocks adds $`\mathrm{ssp}(m_i D_1 + d_1) D_2 + d_2`$,
    where $`m_i`$ sums $`f_{ij} x_j A`$ over the pairs within the cutoff $`r_c`$, each pair
    sending to both its atoms. The filter is

    ```math
    f_{ij} = (\mathrm{ssp}(g(r_{ij}) B_1 + b_1) B_2 + b_2) C(r_{ij}),
    ```

    with K = ``gaussians`` fixed Gaussians $`g_k(r) = \exp(-(r - kD)^2 / (2D^2))`$,
    $`D = r_c / (K - 1)`$, and the envelope $`C(r) = (1 + \cos(\pi r / r_c)) / 2`$, so a pair's
    contribution to the energy and forces vanishes smoothly at the cutoff. The energy is
    $`E = \sum_i \mathrm{ssp}(x_i R_1 + e_1) R_2 + e_2`$ in kJ/mol, with
    $`\mathrm{ssp}(x) = \ln(1 + e^x) - \ln 2`$.

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


def angular_schnet(
    *,
    atom_types: Tensor,
    num_types: int,
    cutoff: Real,
    features: int = 64,
    interactions: int = 3,
    gaussians: int = 50,
    channels: int = 8,
    orders: int = 2,
    trainable: bool = False,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Potential:
    r"""Build a SchNet network whose atoms also start from angle features of their neighbors.

    The network is ``schnet``'s, with each atom's starting features

    ```math
    x_i = \mathrm{table}[t_i] + \mathrm{ssp}(f_i A_1 + a_1) A_2 + a_2.
    ```

    The features come from M = ``channels`` learned radial channels
    $`R_m(r) = C(r) \sum_k g_k(r) W_{km}`$, with the envelope C and the Gaussians $`g_k`$ of
    ``schnet``. Over atom i's neighbors j within the cutoff, with $`u_{ij}`$ the unit vector of
    $`r_i - r_j`$, the sums

    ```math
    S_{im} = \sum_j R_m(r_{ij}), \quad V_{im} = \sum_j R_m(r_{ij}) u_{ij},
    \quad Q_{im} = \sum_j R_m(r_{ij}) (u_{ij} u_{ij}^T - I/3),
    \quad T_{iml} = \sum_j R_m(r_{ij}) R_l(r_{ij}),
    ```

    and the smooth neighbor count $`n_i = \sum_j C(r_{ij})`$, give the $`M + M(M + 1)`$
    features $`s_{im} = S_{im} / (1 + n_i)`$ and, for $`m \le l`$,

    ```math
    (V_{im} \cdot V_{il} - T_{iml}) / D_{iml}, \quad
    (\mathrm{Tr}(Q_{im} Q_{il}) - 2 T_{iml} / 3) / D_{iml}, \quad
    D_{iml} = \sqrt{(T_{imm} + 10^{-4})(T_{ill} + 10^{-4})}.
    ```

    These are $`\sum_{j \ne k} R_m(r_{ij}) R_l(r_{ik}) \cos\theta_{jik}`$ and the same with
    $`\cos^2\theta_{jik} - 1/3`$, divided by D: three-body correlations of the neighbor
    directions, which a distance-only network does not see directly, near 0 where those
    directions have no order, dimensionless, and for sums T well above $`10^{-4}`$ unchanged by
    the scale of the radial channels. They need no calibration.

    With ``orders`` L above 2, each order n from 3 to L adds, for $`m \le l`$, the
    $`M(M + 1)/2`$ features

    ```math
    \sum_{j \ne k} R_m(r_{ij}) R_l(r_{ik}) P_n(\cos\theta_{jik}) / D_{iml},
    ```

    with $`P_n`$ the Legendre polynomial: the moments of the angle distribution that orders 1
    and 2 above do not hold. With the real harmonics $`C_n`$ of $`2n + 1`$ components,
    normalized so that $`C_n(u) \cdot C_n(v) = P_n(u \cdot v)`$, atom i sums
    $`A_{imn} = \sum_j R_m(r_{ij}) C_n(u_{ij})`$, and each feature is
    $`(A_{imn} \cdot A_{iln} - T_{iml}) / D_{iml}`$, so no triplets are formed. Only the first
    angle layer's input grows, to $`M + L M(M + 1)/2`$.

    The arguments are ``schnet``'s, ``channels``, an integer of at least 1, and ``orders``,
    an integer of at least 2, whose default is the network above. Weights are drawn in
    float64 and cast to ``dtype``: the SchNet part exactly as ``schnet`` draws it from
    ``seed``, the angle layers from ``seed`` + 1 (matrices Xavier-uniform, biases 0). The
    state dict holds only the weights. Syncs with the host.

    Raises ``ValueError`` as ``schnet`` does, for ``channels`` that is not an integer of at
    least 1, and for ``orders`` that is not an integer of at least 2.
    """
    return AngularSchNet(
        atom_types=atom_types,
        num_types=num_types,
        cutoff=cutoff,
        features=features,
        interactions=interactions,
        gaussians=gaussians,
        channels=channels,
        orders=orders,
        trainable=trainable,
        seed=seed,
        dtype=dtype,
    )


def sum(*models: Potential) -> Potential:
    r"""Build a model whose energy is the sum of the models' energies.

    $`E = \sum_k E_k`$ over two or more ``models``, so its forces and virial are the sums of
    theirs. Its cutoff is the largest of theirs. MD and fit give each model a neighbor list
    at its own cutoff, so a short-range model beside a long-range one costs about what it
    costs alone; a list without inner lists (see ``neighbors.NeighborList``) serves every
    model, at the cost of the longest. The models must share their atoms, dtype and device.

    The sum holds the models themselves, not copies: its parameters are theirs, named
    ``parts.<k>.<name>``, k counting the models from 0. Each trains as it was built to, so
    fitting a held prior plus a trainable network fits the network to the forces the prior
    misses, and training the sum trains the models in place. A model in two sums shares its
    parameters. Convert a sum with ``to`` as a whole: a model converted alone no longer
    matches the others. No host sync.

    Raises ``TypeError`` for a model that is not a Potential, and ``ValueError`` for fewer
    than two models, or for models of other atom counts, dtypes or devices, when the sum is
    built or, after one model was converted alone, used.
    """
    return Sum(*models)


def tabulated(
    *,
    atom_types: Tensor,
    table: Tensor,
    start: Tensor,
    cutoff: Real,
    trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
) -> Potential:
    r"""Build a pair potential tabulated per unordered type pair, 0 at the cutoff.

    E sums u(r) over the pairs closer than the cutoff $`r_c`$, each reading the row of its two
    types. Between knots u is a $`C^1`$ Catmull-Rom cubic, exact for any quadratic; below a
    row's first knot u follows its tangent. Rows end at 0 at the cutoff, so the energy is
    continuous there; the force steps to zero.

    ``atom_types`` [N], int64 in [0, T), fixes the atoms. ``table`` [P, K], $`P = T(T+1)/2`$
    and $`K \ge 4`$, holds u in kJ/mol at each row's knots, one row per unordered type pair in
    the order (0, 0), (0, 1), ..., (1, 1), ...; its last column, at the cutoff, must be 0. Row
    p's knots lie at $`\mathrm{start}_p + k h_p`$, $`h_p = (r_c - \mathrm{start}_p) / (K - 1)`$,
    with ``start`` [P] and ``cutoff`` in nm. ``trainable`` is ("table",) to train every row or
    {"table": rows} to train only those; no fit moves the last knot. The model computes in the
    dtype and on the device of ``table``. Syncs with the host.

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


__all__ = ("angular_schnet", "lennard_jones", "opls", "schnet", "sum", "tabulated")
