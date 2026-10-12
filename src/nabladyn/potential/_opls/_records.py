"""The records an OPLS model is built from: its topology and its parameter tables."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, kw_only=True, eq=False)
class Topology:
    """Which atoms interact, and which parameter row each atom and each interaction reads.

    All fields are int64 tensors, for N atoms, B bonds, A angles and D dihedrals:

    - ``charge_types`` [N]: each atom's row of ``Parameters.charge``;
    - ``lj_types`` [N]: each atom's row of ``Parameters.sigma`` and ``Parameters.epsilon``;
    - ``bonds`` [B, 2] and ``bond_types`` [B]: each bond's row of ``bond_r0`` and ``bond_k``;
    - ``angles`` [A, 3], the vertex in the middle, and ``angle_types`` [A];
    - ``dihedrals`` [D, 4], propers and impropers in the force field's atom order, and
      ``dihedral_types`` [D]; a dihedral with several Fourier terms takes one row per term.

    The bonds fix the pair classes by the shortest bond path: atoms 1 or 2 bonds apart do
    not interact; atoms 3 bonds apart are 1-4 pairs, scaled by 0.5.
    """

    charge_types: Tensor
    lj_types: Tensor
    bonds: Tensor
    bond_types: Tensor
    angles: Tensor
    angle_types: Tensor
    dihedrals: Tensor
    dihedral_types: Tensor


@dataclass(frozen=True, kw_only=True, eq=False)
class Parameters:
    r"""The force field's parameter tables, one row per type of the ``Topology``.

    Vectors of one float dtype, float32 or float64, on one device, except
    ``dihedral_periodicity``, int64 on that device:

    - ``charge`` [Tq] in e;
    - ``sigma`` [Tlj] in nm and ``epsilon`` [Tlj] in kJ/mol, both >= 0, mixed geometrically
      for each pair; a type with epsilon 0 has no Lennard-Jones term;
    - ``bond_r0`` [Tb] in nm and ``bond_k`` [Tb] in kJ/mol/nm^2: $`0.5 k (r - r_0)^2`$;
    - ``angle_theta0`` [Ta] in rad and ``angle_k`` [Ta] in kJ/mol/rad^2:
      $`0.5 k (\theta - \theta_0)^2`$;
    - ``dihedral_k`` [Td] in kJ/mol, ``dihedral_periodicity`` [Td], $`n \ge 1`$, and
      ``dihedral_phase`` [Td] in rad: $`k [1 + \cos(n\phi - \mathrm{phase})]`$.
    """

    charge: Tensor
    sigma: Tensor
    epsilon: Tensor
    bond_r0: Tensor
    bond_k: Tensor
    angle_theta0: Tensor
    angle_k: Tensor
    dihedral_k: Tensor
    dihedral_periodicity: Tensor
    dihedral_phase: Tensor


__all__ = ("Parameters", "Topology")
