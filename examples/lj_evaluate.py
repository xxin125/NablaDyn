"""Energy, forces, virial and pressure of a Lennard-Jones argon fluid.

The equilibrated frame holds 512 argon atoms at rho* = 0.8 and T* = 1.0 (119.8 K):
positions and cell in nm, velocities in nm/ps.
"""

from pathlib import Path

import numpy as np
import torch

from nabladyn import neighbors, observables, potential, quantity, units

FRAME = Path(__file__).resolve().parent / "data" / "argon_512.npz"


def main():
    dtype = torch.float64
    device = "cuda"
    sigma = 0.34  # nm
    epsilon = 0.996  # kJ/mol
    mass = 39.948  # Da
    cutoff = 1.0  # nm; the energy is shifted to zero there
    skin = 0.1  # nm

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
    neighbor_list = neighbors.NeighborList(cutoff, skin=skin)
    neighbor_list.allocate(positions, cell)

    evaluate = quantity.energy_force_virial(model)
    energy, forces, virial = evaluate(positions, cell, neighbor_list.tensors())

    # The pressure from the virial and the instantaneous kinetic energy.
    kinetic = observables.kinetic_energy(velocities, masses)
    pressure = observables.pressure(kinetic, virial, cell)  # kJ/mol/nm^3

    print(f"Pairs within {neighbor_list.radius} nm: {neighbor_list.num_pairs}")
    print(f"Energy: {float(energy):.2f} kJ/mol")
    print(f"Largest force: {float(forces.norm(dim=1).max()):.2f} kJ/mol/nm")
    print(f"Virial (kJ/mol):\n{virial.cpu().numpy()}")
    print(f"Pressure: {float(pressure) / units.BAR:.0f} bar")
    return energy, forces, virial


if __name__ == "__main__":
    main()
