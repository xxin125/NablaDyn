"""The OPLS-AA model of a fixed topology: bonded terms, Lennard-Jones and Coulomb by PME."""

import math
from numbers import Real

import torch
from torch import Tensor, nn

from ...neighbors import NeighborPairs
from .._base import Potential
from .._lennard_jones._potential import _check_cutoff
from ._bonded import angle_energy, bond_energy, dihedral_energy
from ._pairs import (
    excluded_energy,
    geometric_rows,
    one_four_energy,
    ordinary_energy,
    pair_classes,
    partner_table,
)
from ._pme import influence_table, mesh_energy, plan, signed_modes
from ._records import Parameters, Topology

_TABLES = (
    "charge",
    "sigma",
    "epsilon",
    "bond_r0",
    "bond_k",
    "angle_theta0",
    "angle_k",
    "dihedral_k",
    "dihedral_phase",
)
_TOPOLOGY = (
    "charge_types",
    "lj_types",
    "bonds",
    "bond_types",
    "angles",
    "angle_types",
    "dihedrals",
    "dihedral_types",
)
# Each interaction list, its atoms per row, its types and a table those types index.
_INTERACTIONS = (
    ("bonds", 2, "bond_types", "bond_r0"),
    ("angles", 3, "angle_types", "angle_theta0"),
    ("dihedrals", 4, "dihedral_types", "dihedral_k"),
)
# Tables read by the same types, which must therefore have one length.
_PAIRED_TABLES = (
    ("sigma", "epsilon"),
    ("bond_r0", "bond_k"),
    ("angle_theta0", "angle_k"),
    ("dihedral_k", "dihedral_periodicity", "dihedral_phase"),
)


def _check_records(topology: Topology, parameters: Parameters) -> None:
    """Raise ``ValueError`` unless the inputs are an ``opls.Topology`` and ``opls.Parameters``."""
    if not isinstance(topology, Topology) or not isinstance(parameters, Parameters):
        raise ValueError("topology and parameters must be an opls.Topology and opls.Parameters")


def _check_tables(parameters: Parameters) -> None:
    """Raise ``ValueError`` unless the tables are vectors on sigma's device, of valid dtypes.

    The float tables share sigma's dtype, float32 or float64; the periodicities are int64.
    Tables read by the same types have one length.
    """
    sigma = parameters.sigma
    for name in _TABLES:
        value = getattr(parameters, name)
        if (
            not isinstance(value, Tensor)
            or value.ndim != 1
            or value.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError(f"parameters.{name} must be a float32 or float64 vector")
        if (value.dtype, value.device) != (sigma.dtype, sigma.device):
            raise ValueError(f"parameters.{name} must share the dtype and device of sigma")
    periodicity = parameters.dihedral_periodicity
    if (
        not isinstance(periodicity, Tensor)
        or periodicity.ndim != 1
        or periodicity.dtype != torch.int64
        or periodicity.device != sigma.device
    ):
        raise ValueError(
            "parameters.dihedral_periodicity must be an int64 vector on the device of sigma",
        )
    for names in _PAIRED_TABLES:
        if len({getattr(parameters, name).numel() for name in names}) != 1:
            raise ValueError(f"parameters {', '.join(names)} must have the same length")


def _check_index(name: str, value: Tensor, shape: tuple[int | None, ...]) -> None:
    """Raise ``ValueError`` unless a topology field is an int64 tensor of this shape.

    None in ``shape`` accepts any length.
    """
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.int64
        or value.ndim != len(shape)
        or any(want is not None and got != want for got, want in zip(value.shape, shape))
    ):
        rendered = ", ".join("*" if size is None else str(size) for size in shape)
        raise ValueError(f"topology.{name} must be an int64 tensor of shape [{rendered}]")


def _check_range(name: str, value: Tensor, count: int, what: str) -> None:
    """Raise ``ValueError`` unless every entry of a topology field lies in [0, count)."""
    if value.numel() and not bool(((value >= 0) & (value < count)).all()):
        raise ValueError(f"topology.{name} must hold {what} in [0, {count})")


def _check_topology(topology: Topology, parameters: Parameters) -> None:
    """Raise ``ValueError`` unless the topology's atoms and types fit each other and the tables.

    Syncs with the host.
    """
    _check_index("lj_types", topology.lj_types, (None,))
    num_atoms = topology.lj_types.numel()
    if num_atoms == 0:
        raise ValueError("topology.lj_types must hold at least one atom")
    _check_index("charge_types", topology.charge_types, (num_atoms,))
    _check_range("charge_types", topology.charge_types, parameters.charge.numel(), "rows")
    _check_range("lj_types", topology.lj_types, parameters.sigma.numel(), "rows")
    for name, width, types, table in _INTERACTIONS:
        index = getattr(topology, name)
        _check_index(name, index, (None, width))
        _check_index(types, getattr(topology, types), (index.shape[0],))
        _check_range(name, index, num_atoms, "atoms")
        _check_range(types, getattr(topology, types), getattr(parameters, table).numel(), "rows")


def _check_planning_cell(cell: Tensor) -> list[float]:
    """Return the planning cell's lengths in nm as floats.

    Raises ``ValueError`` unless the cell is a float tensor [3], finite and > 0. Syncs with
    the host.
    """
    if not isinstance(cell, Tensor) or cell.shape != (3,) or not cell.is_floating_point():
        raise ValueError("cell must be a float tensor of shape [3]")
    lengths = cell.detach().double().cpu().tolist()
    if not all(math.isfinite(length) and length > 0 for length in lengths):
        raise ValueError("cell lengths must be finite and positive")
    return lengths


def _check_ewald_tolerance(tolerance: Real) -> None:
    """Raise ``ValueError`` unless the Ewald tolerance is a real number in (0, 0.5)."""
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not 0 < tolerance < 0.5
    ):
        raise ValueError("ewald_tolerance must be a real number in (0, 0.5)")


def _check_values(parameters: Parameters) -> None:
    """Raise ``ValueError`` unless the tables are finite, sigma and epsilon >= 0 and n >= 1.

    Syncs with the host.
    """
    tables = torch.cat([getattr(parameters, name) for name in _TABLES])
    if not bool(tables.isfinite().all()):
        raise ValueError("the parameter tables must be finite")
    if not bool(((parameters.sigma >= 0) & (parameters.epsilon >= 0)).all()):
        raise ValueError("sigma and epsilon must be >= 0")
    if not bool((parameters.dihedral_periodicity >= 1).all()):
        raise ValueError("dihedral_periodicity must be >= 1")


class OPLS(Potential):
    """OPLS-AA of a fixed topology: bonds, angles, dihedrals, Lennard-Jones and Coulomb.

    Built by ``potential.opls``, which documents the terms, arguments, units and errors.
    The tables are frozen parameters and the topology fields buffers, named as the fields
    of ``opls.Parameters`` and ``opls.Topology``.
    """

    def __init__(
        self,
        *,
        topology: Topology,
        parameters: Parameters,
        cutoff: Real,
        cell: Tensor,
        ewald_tolerance: Real = 5e-4,
    ):
        super().__init__()
        _check_records(topology, parameters)
        _check_cutoff(cutoff)
        lengths = _check_planning_cell(cell)
        _check_ewald_tolerance(ewald_tolerance)
        _check_tables(parameters)
        _check_topology(topology, parameters)
        _check_values(parameters)
        device, dtype = parameters.sigma.device, parameters.sigma.dtype
        num_atoms = topology.lj_types.numel()
        self._cutoff = float(cutoff)
        self._num_lj_types = parameters.sigma.numel()
        self._alpha, self._mesh = plan(self._cutoff, float(ewald_tolerance), lengths)
        for name in _TABLES:
            value = getattr(parameters, name).detach().clone()
            self.register_parameter(name, nn.Parameter(value, requires_grad=False))
        self.register_buffer("dihedral_periodicity", parameters.dihedral_periodicity.clone())
        for name in _TOPOLOGY:
            self.register_buffer(name, getattr(topology, name).detach().to(device, copy=True))
        excluded, one_four = pair_classes(topology.bonds.cpu(), num_atoms)
        partners = partner_table(excluded, one_four, num_atoms)
        self.register_buffer("_excluded", excluded.to(device), persistent=False)
        self.register_buffer("_one_four", one_four.to(device), persistent=False)
        self.register_buffer("_partners", partners.to(device), persistent=False)
        # Row-major upper triangle, the order of pair_index.
        mix = torch.triu_indices(self._num_lj_types, self._num_lj_types, device=device)
        self.register_buffer("_mix", mix, persistent=False)
        influence = influence_table(self._mesh, dtype, device)
        self.register_buffer("_influence", influence, persistent=False)
        for axis, modes in enumerate(signed_modes(self._mesh, dtype, device)):
            self.register_buffer(f"_modes_{axis}", modes, persistent=False)

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def num_atoms(self) -> int:
        return self.lj_types.numel()

    @property
    def dtype(self) -> torch.dtype:
        return self.sigma.dtype

    @property
    def device(self) -> torch.device:
        return self.sigma.device

    @property
    def alpha(self) -> float:
        """The Ewald splitting parameter in nm^-1, fixed at construction."""
        return self._alpha

    @property
    def mesh(self) -> tuple[int, int, int]:
        """The PME mesh (K1, K2, K3), fixed at construction."""
        return self._mesh

    def _energy_parts(
        self,
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return the energy of each part in kJ/mol, for positions already in [0, L).

        The parts, "bonds", "angles", "dihedrals", "lj", "lj_14", "coulomb", "coulomb_14",
        "coulomb_excluded" and "mesh", sum to the energy.
        """
        charges = self.charge[self.charge_types]
        sigma_rows = geometric_rows(self.sigma, self._mix[0], self._mix[1])
        epsilon_rows = geometric_rows(self.epsilon, self._mix[0], self._mix[1])
        bonds = bond_energy(
            positions,
            cell,
            strain,
            self.bonds,
            self.bond_r0[self.bond_types],
            self.bond_k[self.bond_types],
        )
        angles = angle_energy(
            positions,
            cell,
            strain,
            self.angles,
            self.angle_theta0[self.angle_types],
            self.angle_k[self.angle_types],
        )
        dihedrals = dihedral_energy(
            positions,
            cell,
            strain,
            self.dihedrals,
            self.dihedral_k[self.dihedral_types],
            self.dihedral_periodicity[self.dihedral_types],
            self.dihedral_phase[self.dihedral_types],
        )
        lennard_jones, coulomb = ordinary_energy(
            positions,
            cell,
            strain,
            pairs.index,
            self._partners,
            self._cutoff,
            self.lj_types,
            self._num_lj_types,
            sigma_rows,
            epsilon_rows,
            charges,
            self._alpha,
        )
        lennard_jones_14, coulomb_14 = one_four_energy(
            positions,
            cell,
            strain,
            self._one_four,
            self.lj_types,
            self._num_lj_types,
            sigma_rows,
            epsilon_rows,
            charges,
            self._alpha,
        )
        mesh = mesh_energy(
            charges,
            positions,
            cell,
            strain,
            self._alpha,
            self._mesh,
            self._influence,
            (self._modes_0, self._modes_1, self._modes_2),
        )
        return {
            "bonds": bonds,
            "angles": angles,
            "dihedrals": dihedrals,
            "lj": lennard_jones,
            "lj_14": lennard_jones_14,
            "coulomb": coulomb,
            "coulomb_14": coulomb_14,
            "coulomb_excluded": excluded_energy(
                positions,
                cell,
                strain,
                self._excluded,
                charges,
                self._alpha,
            ),
            "mesh": mesh,
        }

    def _energy_wrapped(
        self,
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> Tensor:
        return sum(self._energy_parts(positions, cell, pairs, strain).values())


__all__ = ("OPLS",)
