"""Run the SchNet model of coarse-grained methanol in NVT at three densities.

Loads ``cg_schnet.pt`` from ``cg_schnet_fit.py`` and runs 100 ps of NVT at 298.15 K at each
density of its training data, 454.98, 473.54 and 492.85 Da/nm^3, from the last frame of that
density's file. It compares the beads' g(r) with the all-atom one.
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from nabladyn import md, neighbors, observables, potential

DATA = Path(__file__).resolve().parent / "data"
FILES = ("methanol_cg_nvt_low.npz", "methanol_cg_nvt.npz", "methanol_cg_nvt_high.npz")


def main():
    dtype = torch.float32
    device = "cuda"
    cutoff = 1.0  # nm, as in training
    temperature = 298.15  # K
    seed = 21  # of the starting velocities
    timestep = 0.004  # ps
    equilibration = 5000  # steps
    steps = 25000  # steps sampled per density, 100 ps
    sample_every = 100  # steps
    weights_file = "cg_schnet.pt"
    figure_file = "cg_schnet_nvt.png"

    sets = [np.load(DATA / name) for name in FILES]
    model = potential.schnet(
        atom_types=torch.as_tensor(sets[0]["atom_types"], device=device),
        num_types=1,
        cutoff=cutoff,
        dtype=dtype,
    )
    model.load_state_dict(torch.load(weights_file, map_location=device))

    # NVT at each density; the g(r) from the sampled frames.
    results = []
    for index, data in enumerate(sets):
        masses = torch.tensor(data["masses"], dtype=dtype, device=device)
        cell = torch.tensor(data["cells"][-1], dtype=dtype, device=device)
        simulation = md.nvt(
            model,
            torch.tensor(data["positions"][-1], dtype=dtype, device=device),
            md.maxwell_boltzmann(masses, temperature, seed=seed),
            masses,
            cell,
            timestep=timestep,
            temperature=temperature,
            thermostat_timescale=0.1,  # ps
        )
        simulation.run(equilibration)
        sample_file = Path(f"cg_schnet_nvt_{index}.h5")
        simulation.run(steps, sample_every=sample_every, sample_file=sample_file)
        stored, _ = md.read_sample_file(sample_file)
        edges = torch.tensor(data["edges"], dtype=dtype, device=device)
        rdf = 0.0
        for positions in stored["positions"]:
            frame = torch.tensor(positions, dtype=dtype, device=device)
            neighbor_list = neighbors.NeighborList(cutoff)
            neighbor_list.allocate(frame, cell)
            rdf = rdf + observables.rdf(frame, cell, neighbor_list.tensors(), edges)[0]
        rdf = (rdf / len(stored["positions"])).cpu().numpy()
        error = float(np.sqrt(np.mean((rdf - data["rdf"][0]) ** 2)))
        print(f"{float(data['density']):.2f} Da/nm^3: g(r) RMS deviation {error:.4f}")
        results.append((float(data["density"]), rdf, error))

    figure = Figure(figsize=(13.0, 4.0), layout="constrained")
    rdf_axes = figure.subplots(1, 3)
    centres = 0.5 * (sets[0]["edges"][:-1] + sets[0]["edges"][1:])
    for axes, data, (value, rdf, error) in zip(rdf_axes, sets, results):
        axes.plot(centres, data["rdf"][0], color="black", label="all-atom")
        axes.plot(centres, rdf, "--", color="tab:orange", label=f"SchNet ({error:.4f})")
        axes.set_title(f"{value:.2f} Da/nm^3", fontsize=10)
        axes.set_xlabel("r (nm)")
        axes.legend(fontsize=8)
    rdf_axes[0].set_ylabel("g(r) of the beads")
    figure.savefig(figure_file, dpi=150)
    print(f"Wrote {figure_file}")
    return results


if __name__ == "__main__":
    main()
