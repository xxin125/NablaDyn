"""NVT molecular dynamics of liquid methanol with OPLS-AA and particle-mesh Ewald.

``data/methanol_opls.npz`` holds 343 methanol molecules (2,058 atoms) after 400 ps of
NPT at 298.15 K and 1 bar with this force field: positions and cell in nm, velocities
in nm/ps, masses in Da, and the OPLS-AA topology and tables under the fields of
``potential.opls.Topology`` and ``potential.opls.Parameters``. The run keeps the cell and
holds 298.15 K with a Nose-Hoover chain in float32 with flexible bonds at 0.5 fs, prints the
panel as it goes and stores every sample in an HDF5 file.
"""

from pathlib import Path

import numpy as np
import torch

from nabladyn import md, potential, units

DATA = Path(__file__).resolve().parent / "data" / "methanol_opls.npz"


def main():
    dtype = torch.float32
    device = "cuda"
    cutoff = 1.0  # nm, for Lennard-Jones and the real-space Coulomb, both shifted there
    ewald_tolerance = 5e-4  # about the mesh's relative force error
    temperature = 298.15  # K
    timestep = 0.0005  # ps, short enough for the flexible X-H bonds (periods near 10 fs)
    steps = 20000  # 10 ps
    sample_every = 800  # steps, 0.4 ps
    sample_file = Path("opls_nvt.h5")

    data = np.load(DATA)

    def indices(name):
        return torch.as_tensor(data[name], device=device)

    def values(name):
        return torch.as_tensor(data[name], dtype=dtype, device=device)

    # Which atoms interact, and which table row every atom and every term reads.
    topology = potential.opls.Topology(
        charge_types=indices("charge_types"),
        lj_types=indices("lj_types"),
        bonds=indices("bonds"),
        bond_types=indices("bond_types"),
        angles=indices("angles"),
        angle_types=indices("angle_types"),
        dihedrals=indices("dihedrals"),
        dihedral_types=indices("dihedral_types"),
    )
    # OPLS-AA's values for methanol's types, in e, nm, kJ/mol and rad.
    parameters = potential.opls.Parameters(
        charge=values("charge"),
        sigma=values("sigma"),
        epsilon=values("epsilon"),
        bond_r0=values("bond_r0"),
        bond_k=values("bond_k"),
        angle_theta0=values("angle_theta0"),
        angle_k=values("angle_k"),
        dihedral_k=values("dihedral_k"),
        dihedral_periodicity=indices("dihedral_periodicity"),
        dihedral_phase=values("dihedral_phase"),
    )
    positions, velocities = values("positions"), values("velocities")
    masses, cell = values("masses"), values("cell")

    # The mesh is planned for this cell, which NVT keeps.
    model = potential.opls(
        topology=topology,
        parameters=parameters,
        cutoff=cutoff,
        cell=cell,
        ewald_tolerance=ewald_tolerance,
    )
    print(f"Ewald: alpha {model.alpha:.4f} nm^-1, mesh {model.mesh}")
    simulation = md.nvt(
        model,
        positions,
        velocities,
        masses,
        cell,
        timestep=timestep,
        temperature=temperature,
        thermostat_timescale=0.1,  # ps
    )
    simulation.run(
        steps,
        sample_every=sample_every,
        echo=True,
        sample_file=sample_file,
    )

    samples, settings = md.read_sample_file(sample_file)
    print(f"{settings['ensemble']} at {settings['temperature']} K, {len(samples['step'])} samples")
    print(f"Mean temperature: {samples['temperature'].mean():.1f} K")
    # The flexible bonds make each sample's pressure scatter by about 1,000 bar, so the mean
    # of 25 samples is good to about 200 bar.
    print(f"Mean pressure: {samples['pressure'].mean() / units.BAR:.0f} bar")
    return samples


if __name__ == "__main__":
    main()
