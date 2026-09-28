"""NVT molecular dynamics of a Lennard-Jones argon fluid, with a sample file.

The equilibrated frame holds 512 argon atoms at rho* = 0.8 and T* = 1.0 (119.8 K):
positions and cell in nm, velocities in nm/ps. The run keeps it at 120 K with a
Nose-Hoover chain, prints the panel as it goes and stores every sample in an HDF5 file.
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
    temperature = 120.0  # K
    timestep = 0.004  # ps
    steps = 2500  # 10 ps
    sample_every = 100  # steps
    sample_file = Path("lj_nvt.h5")

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
    print(f"Mean pressure: {samples['pressure'].mean() / units.BAR:.0f} bar")
    return samples


if __name__ == "__main__":
    main()
