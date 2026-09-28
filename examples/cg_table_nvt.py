"""Run a pair table of coarse-grained methanol in NVT at three densities.

Loads a table saved by ``cg_table_fit.py`` (force matching) or ``cg_table_ibi.py`` (Iterative
Boltzmann Inversion at the experimental density) and runs 100 ps of NVT at 298.15 K at each
density of the data, 454.98, 473.54 and 492.85 Da/nm^3, from the last frame of that
density's file. It compares the beads' g(r) with the all-atom one; IBI fits one density
only.

    python examples/cg_table_nvt.py [cg_table.pt | cg_table_ibi.pt]
"""

import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from nabladyn import md, neighbors, observables, potential

DATA = Path(__file__).resolve().parent / "data"
FILES = ("methanol_cg_nvt_low.npz", "methanol_cg_nvt.npz", "methanol_cg_nvt_high.npz")


def main(table_file="cg_table.pt"):
    dtype = torch.float32
    device = "cuda"
    cutoff = 1.0  # nm, as in training
    temperature = 298.15  # K
    seed = 21  # of the starting velocities
    timestep = 0.004  # ps
    equilibration = 5000  # steps
    steps = 25000  # steps sampled per density, 100 ps
    sample_every = 100  # steps
    label = Path(table_file).stem
    figure_file = f"{label}_nvt.png"

    sets = [np.load(DATA / name) for name in FILES]
    saved = torch.load(table_file, map_location=device)
    model = potential.tabulated(
        atom_types=torch.as_tensor(sets[0]["atom_types"], device=device),
        table=saved["table"].to(dtype),
        start=saved["start"].to(dtype),
        cutoff=cutoff,
    )

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
        sample_file = Path(f"{label}_nvt_{index}.h5")
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
        axes.plot(centres, rdf, "--", color="tab:green", label=f"{label} ({error:.4f})")
        axes.set_title(f"{value:.2f} Da/nm^3", fontsize=10)
        axes.set_xlabel("r (nm)")
        axes.legend(fontsize=8)
    rdf_axes[0].set_ylabel("g(r) of the beads")
    figure.savefig(figure_file, dpi=150)
    print(f"Wrote {figure_file}")
    return model, results


if __name__ == "__main__":
    main(*sys.argv[1:])
