"""Fit a pair table for coarse-grained methanol by Iterative Boltzmann Inversion.

``data/methanol_cg_nvt.npz`` maps an all-atom NVT run of 343 methanol molecules at 298.15 K
and the experimental density, 473.54 Da/nm^3, to one bead per molecule, with the beads' g(r)
on 100 bins to 1 nm. IBI starts from u(r) = -kT ln g*(r), the target g(r), and repeats:
simulate the beads with the current table, measure g(r), and correct
u(r) <- u(r) + kT ln(g(r) / g*(r)) wherever both are resolved. It needs no forces, only the
structure, and reproduces the target g(r). The table holds u(r) at knots 0.01 nm apart from
0.29 nm, where the data resolve the wall, to the cutoff. It is saved for ``cg_table_nvt.py``,
which runs it at three densities.
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from nabladyn import md, neighbors, observables, potential, units

DATA = Path(__file__).resolve().parent / "data" / "methanol_cg_nvt.npz"


def main():
    dtype = torch.float32
    device = "cuda"
    cutoff = 1.0  # nm
    start = 0.29  # nm
    knots = 72  # 0.01 nm apart
    temperature = 298.15  # K
    seed = 21  # of the starting velocities
    timestep = 0.004  # ps
    iterations = 30
    equilibration = 2500  # steps after each correction, 10 ps
    steps = 12500  # steps sampled per iteration, 50 ps
    sample_every = 50  # steps
    resolved = 1e-2  # the smallest g(r) a correction uses
    sample_file = Path("cg_table_ibi.h5")  # each iteration's samples, overwritten
    table_file = "cg_table_ibi.pt"
    figure_file = "cg_table_ibi.png"

    data = np.load(DATA)
    edges = data["edges"]
    centres = 0.5 * (edges[:-1] + edges[1:])
    target = data["rdf"][0]
    r = np.linspace(start, cutoff, knots)
    kT = units.BOLTZMANN * temperature

    def on_knots(g):
        return np.interp(r, centres, g)

    table = -kT * np.log(np.maximum(on_knots(target), resolved))
    table -= table[-1]
    first = table.copy()
    model = potential.tabulated(
        atom_types=torch.as_tensor(data["atom_types"], device=device),
        table=torch.tensor(table[None], dtype=dtype, device=device),
        start=torch.tensor([start], dtype=dtype, device=device),
        cutoff=cutoff,
    )
    masses = torch.tensor(data["masses"], dtype=dtype, device=device)
    cell = torch.tensor(data["cells"][-1], dtype=dtype, device=device)
    # The simulation reads the table in place, so one simulation serves every iteration.
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
    edges_tensor = torch.tensor(edges, dtype=dtype, device=device)
    errors = []
    for iteration in range(iterations):
        simulation.run(equilibration)
        simulation.run(steps, sample_every=sample_every, sample_file=sample_file)
        stored, _ = md.read_sample_file(sample_file)
        g = 0.0
        for positions in stored["positions"]:
            frame = torch.tensor(positions, dtype=dtype, device=device)
            neighbor_list = neighbors.NeighborList(cutoff)
            neighbor_list.allocate(frame, cell)
            g = g + observables.rdf(frame, cell, neighbor_list.tensors(), edges_tensor)[0]
        g = (g / len(stored["positions"])).cpu().numpy()
        errors.append(float(np.sqrt(np.mean((g - target) ** 2))))
        print(f"iteration {iteration:2d}: g(r) RMS deviation {errors[-1]:.4f}")
        if iteration == iterations - 1:
            break
        current, wanted = on_knots(g), on_knots(target)
        use = (current > resolved) & (wanted > resolved)
        table[use] += kT * np.log(current[use] / wanted[use])
        table -= table[-1]
        with torch.no_grad():
            model.table.copy_(torch.tensor(table[None], dtype=dtype, device=device))

    torch.save(model.state_dict(), table_file)

    figure = Figure(figsize=(13.0, 4.2), layout="constrained")
    error_axes, table_axes, rdf_axes = figure.subplots(1, 3)
    error_axes.semilogy(np.arange(iterations), errors, "o-")
    error_axes.set_xlabel("iteration")
    error_axes.set_ylabel("g(r) RMS deviation")
    table_axes.plot(r, first, ":", color="gray", label="start: -kT ln g*(r)")
    table_axes.plot(r, table, color="tab:purple", label="after IBI")
    table_axes.set_ylim(-3.0, 6.0)
    table_axes.set_xlabel("r (nm)")
    table_axes.set_ylabel("u(r) (kJ/mol)")
    table_axes.legend()
    rdf_axes.plot(centres, target, color="black", label="all-atom")
    rdf_axes.plot(centres, g, "--", color="tab:purple", label=f"IBI table ({errors[-1]:.4f})")
    rdf_axes.set_xlabel("r (nm)")
    rdf_axes.set_ylabel("g(r) of the beads")
    rdf_axes.legend()
    figure.savefig(figure_file, dpi=150)
    print(f"Wrote {table_file} and {figure_file}")
    return model, errors


if __name__ == "__main__":
    main()
