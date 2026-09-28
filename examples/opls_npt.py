"""NPT molecular dynamics of liquid methanol with OPLS-AA and particle-mesh Ewald.

``data/methanol_opls.npz`` holds 343 methanol molecules (2,058 atoms) after 400 ps of
NPT at 298.15 K and 1 bar with this force field: positions and cell in nm, velocities
in nm/ps, masses in Da, and the OPLS-AA topology and tables under the fields of
``potential.opls.Topology`` and ``potential.opls.Parameters``. The run continues at
298.15 K and 1 bar in float32 with flexible bonds at 0.5 fs, samples the panel and writes
the samples to a file, then prints the mean temperature, pressure and density.
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
    pressure = 1.0 * units.BAR  # kJ/mol/nm^3
    timestep = 0.0005  # ps, short enough for the flexible X-H bonds (periods near 10 fs)
    equilibration_steps = 10000  # 5 ps
    steps = 80000  # 40 ps
    sample_every = 800  # steps, 0.4 ps
    sample_file = Path("opls_npt.h5")

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

    # The mesh is planned for this cell and stays fixed; the box fluctuates by about 0.5%.
    model = potential.opls(
        topology=topology,
        parameters=parameters,
        cutoff=cutoff,
        cell=cell,
        ewald_tolerance=ewald_tolerance,
    )
    print(f"Ewald: alpha {model.alpha:.4f} nm^-1, mesh {model.mesh}")
    simulation = md.npt(
        model,
        positions,
        velocities,
        masses,
        cell,
        timestep=timestep,
        temperature=temperature,
        pressure=pressure,
        thermostat_timescale=0.1,  # ps
        barostat_timescale=1.0,  # ps
    )
    simulation.run(equilibration_steps)
    samples = simulation.run(steps, sample_every=sample_every, echo=True, sample_file=sample_file)

    density = masses.sum().item() / samples["volume"]  # Da/nm^3, one value per sample
    print(f"Mean temperature: {samples['temperature'].mean():.1f} K")
    print(f"Mean pressure: {samples['pressure'].mean() / units.BAR:.0f} bar")
    print(f"Mean density: {density.mean() / units.G_PER_CM3:.4f} g/cm^3")
    return samples


if __name__ == "__main__":
    main()
