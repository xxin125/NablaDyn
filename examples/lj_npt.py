"""NPT molecular dynamics of liquid argon at 85 K and 1 bar.

The equilibrated frame holds 512 argon atoms at rho* = 0.8 and T* = 1.0 (119.8 K):
positions and cell in nm, velocities in nm/ps. The run cools it to 85 K at 1 bar, where
argon is a liquid, with Nose-Hoover chains on the particles and on the isotropic barostat.
Once the box has settled it samples the panel in memory and prints the mean density.
"""

from pathlib import Path

import numpy as np
import torch

from nabladyn import md, potential, units

FRAME = Path(__file__).resolve().parent / "data" / "argon_512.npz"


def main():
    dtype = torch.float32
    device = "cuda"
    sigma = 0.34  # nm
    epsilon = 0.996  # kJ/mol
    mass = 39.948  # Da
    cutoff = 1.0  # nm; the energy is shifted to zero there
    temperature = 85.0  # K
    pressure = 1.0 * units.BAR  # kJ/mol/nm^3
    timestep = 0.004  # ps
    equilibration_steps = 5000  # 20 ps
    steps = 5000  # 20 ps
    sample_every = 50  # steps

    frame = np.load(FRAME)
    positions = torch.tensor(frame["positions"], dtype=dtype, device=device)
    velocities = torch.tensor(frame["velocities"], dtype=dtype, device=device)
    cell = torch.tensor(frame["cell"], dtype=dtype, device=device)
    masses = torch.full((len(positions),), mass, dtype=dtype, device=device)

    # One atom type, so sigma and epsilon hold one value each.
    model = potential.lennard_jones(
        atom_types=torch.zeros(len(positions), dtype=torch.int64, device=device),
        sigma=torch.tensor([sigma], dtype=dtype, device=device),
        epsilon=torch.tensor([epsilon], dtype=dtype, device=device),
        cutoff=cutoff,
    )
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
    samples = simulation.run(steps, sample_every=sample_every, echo=True)

    density = masses.sum().item() / samples["volume"]  # Da/nm^3, one value per sample
    print(f"Mean temperature: {samples['temperature'].mean():.1f} K")
    print(f"Mean pressure: {samples['pressure'].mean() / units.BAR:.0f} bar")
    print(f"Mean density: {density.mean() / units.G_PER_CM3:.3f} g/cm^3")
    return samples


if __name__ == "__main__":
    main()
