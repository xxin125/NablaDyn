"""Run a pair table of coarse-grained methanol in NVT at three densities.

Loads a table saved by ``cg_table_fit.py`` (force matching) or ``cg_table_ibi.py`` (Iterative
Boltzmann Inversion at the experimental density) and runs 400 ps of NVT at 298.15 K at each
density of the data, 454.98, 473.54 and 492.85 Da/nm^3, from the last frame of that
density's file. It compares the beads' g(r), the distribution of the angle j-i-k between two
neighbors j and k of a bead i (neighbors being closer than the all-atom g(r)'s first minimum)
and the pressure with the all-atom ones; IBI fits one density only.

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
ANGLE_EDGES = np.linspace(0.0, 180.0, 61)  # degrees, 3-degree bins


def first_minimum(data) -> float:
    """Return the all-atom g(r)'s first minimum, within 0.15 nm after its first peak, in nm."""
    centres = 0.5 * (data["edges"][:-1] + data["edges"][1:])
    g = data["rdf"][0]
    peak = int(np.argmax(g))
    window = (centres > centres[peak]) & (centres <= centres[peak] + 0.15)
    return float(centres[window][np.argmin(g[window])])


def angle_distribution(positions, cells, shell, dtype, device) -> np.ndarray:
    """Return the mean first-shell angle distribution of frames [F, N, 3] in cells [F, 3].

    Each frame's ``observables.adf`` on ``ANGLE_EDGES``, neighbors closer than ``shell`` nm,
    averaged over the frames; per degree, of unit area.
    """
    edges = torch.tensor(ANGLE_EDGES, dtype=dtype, device=device)
    total = 0.0
    for x, cell in zip(positions, cells):
        frame = torch.tensor(x, dtype=dtype, device=device)
        box = torch.tensor(cell, dtype=dtype, device=device)
        neighbor_list = neighbors.NeighborList(shell)
        neighbor_list.allocate(frame, box)
        total = total + observables.adf(frame, box, neighbor_list.tensors(), edges, r_max=shell)
    return (total / len(positions)).cpu().numpy()


def main(table_file="cg_table.pt"):
    dtype = torch.float32
    device = "cuda"
    cutoff = 1.0  # nm, as in training
    temperature = 298.15  # K
    seed = 21  # of the starting velocities
    timestep = 0.004  # ps
    equilibration = 5000  # steps
    steps = 100000  # steps sampled per density, 400 ps
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

    # NVT at each density; the g(r), the angle distribution and the pressure from the samples.
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
        samples = simulation.run(steps, sample_every=sample_every, sample_file=sample_file)
        pressure = float(samples["pressure"].mean())
        stored, _ = md.read_sample_file(sample_file)
        edges = torch.tensor(data["edges"], dtype=dtype, device=device)
        rdf = 0.0
        for positions in stored["positions"]:
            frame = torch.tensor(positions, dtype=dtype, device=device)
            neighbor_list = neighbors.NeighborList(cutoff)
            neighbor_list.allocate(frame, cell)
            rdf = rdf + observables.rdf(frame, cell, neighbor_list.tensors(), edges)[0]
        rdf = (rdf / len(stored["positions"])).cpu().numpy()
        rdf_error = float(np.sqrt(np.mean((rdf - data["rdf"][0]) ** 2)))
        shell = first_minimum(data)
        angles = angle_distribution(stored["positions"], stored["cell"], shell, dtype, device)
        reference = angle_distribution(data["positions"], data["cells"], shell, dtype, device)
        angle_error = float(np.sqrt(np.mean((angles - reference) ** 2)))
        density, all_atom = float(data["density"]), float(data["pressure"])
        print(
            f"{density:.2f} Da/nm^3: g(r) RMS deviation {rdf_error:.4f}; angle distribution "
            f"RMS deviation {angle_error:.2e} per degree; pressure {pressure:.1f} kJ/mol/nm^3, "
            f"all-atom {all_atom:.1f} +- {float(data['pressure_block_error']):.1f}",
        )
        results.append(
            (density, rdf, rdf_error, reference, angles, angle_error, pressure, all_atom),
        )

    figure = Figure(figsize=(13.0, 7.5), layout="constrained")
    rdf_axes, angle_axes = figure.subplots(2, 3)
    centres = 0.5 * (sets[0]["edges"][:-1] + sets[0]["edges"][1:])
    angle_centres = 0.5 * (ANGLE_EDGES[:-1] + ANGLE_EDGES[1:])
    for column, (data, result) in enumerate(zip(sets, results)):
        density, rdf, rdf_error, reference, angles, angle_error, pressure, all_atom = result
        axes = rdf_axes[column]
        axes.plot(centres, data["rdf"][0], color="black", label="all-atom")
        axes.plot(centres, rdf, "--", color="tab:green", label=f"{label} ({rdf_error:.4f})")
        axes.set_title(
            f"{density:.2f} Da/nm^3\npressure {pressure:.0f} kJ/mol/nm^3, all-atom {all_atom:.0f}",
            fontsize=10,
        )
        axes.set_xlabel("r (nm)")
        axes.legend(fontsize=8)
        axes = angle_axes[column]
        axes.plot(angle_centres, reference, color="black", label="all-atom")
        axes.plot(
            angle_centres,
            angles,
            "--",
            color="tab:green",
            label=f"{label} ({angle_error:.1e})",
        )
        axes.set_xlabel("angle j-i-k (degrees)")
        axes.legend(fontsize=8)
    rdf_axes[0].set_ylabel("g(r) of the beads")
    angle_axes[0].set_ylabel("first-shell angle distribution (1/degree)")
    figure.savefig(figure_file, dpi=150)
    print(f"Wrote {figure_file}")
    return model, results


if __name__ == "__main__":
    main(*sys.argv[1:])
