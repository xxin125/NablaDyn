"""Train a pair table for coarse-grained methanol by force matching at three densities.

The data are those of ``cg_schnet_fit.py``: three all-atom NVT runs of methanol at 298.15 K,
454.98, 473.54 and 492.85 Da/nm^3, mapped to one bead per molecule, 1,000 frames each. The
table holds the pair energy u(r) at knots 0.01 nm apart from 0.29 nm to the cutoff and
starts from the Boltzmann inversion of the g(r), -kT ln g(r). Force matching fits the bead
forces by least squares, the best fit of the forces a pair potential allows. The table
trains on the first 800 frames of each file, validates on the rest and is saved for
``cg_table_nvt.py``, which runs it.
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from nabladyn import potential, training, units

DATA = Path(__file__).resolve().parent / "data"
FILES = ("methanol_cg_nvt_low.npz", "methanol_cg_nvt.npz", "methanol_cg_nvt_high.npz")


def main():
    dtype = torch.float32
    device = "cuda"
    cutoff = 1.0  # nm
    # Every knot needs data: below its first knot a row follows its tangent, and a knot no
    # pair reaches keeps its start value and can bend that tangent into an attractive core.
    start = 0.29  # nm, where the data resolve the wall; the closest pair is 0.276 nm
    knots = 72  # 0.01 nm apart
    temperature = 298.15  # K, of the data
    train_frames = 800  # of each file; the rest validate
    updates = 5000
    batch_size = 8  # frames
    learning_rate = 1e-2  # kJ/mol per knot and step at most
    halve_every = 1000  # updates
    echo_every = 500  # updates
    table_file = "cg_table.pt"
    figure_file = "cg_table_fit.png"

    sets = [np.load(DATA / name) for name in FILES]

    def frames(part):
        fields = ("positions", "cells", "forces")
        joined = {name: np.concatenate([data[name][part] for data in sets]) for name in fields}
        return training.Dataset(joined["positions"], joined["cells"], forces=joined["forces"])

    # Boltzmann inversion of the experimental density's g(r), 0 at the cutoff.
    reference = sets[1]
    centres = 0.5 * (reference["edges"][:-1] + reference["edges"][1:])
    r = np.linspace(start, cutoff, knots)
    kT = units.BOLTZMANN * temperature
    guess = -kT * np.log(np.maximum(np.interp(r, centres, reference["rdf"][0]), 1e-3))
    guess -= guess[-1]
    model = potential.tabulated(
        atom_types=torch.as_tensor(reference["atom_types"], device=device),
        table=torch.tensor(guess[None], dtype=dtype, device=device),
        start=torch.tensor([start], dtype=dtype, device=device),
        cutoff=cutoff,
        trainable=("table",),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=halve_every, gamma=0.5)
    report = training.fit(
        model,
        frames(slice(0, train_frames)),
        optimizer,
        updates=updates,
        batch_size=batch_size,
        forces=1.0,
        validation=frames(slice(train_frames, None)),
        scheduler=scheduler,
        echo_every=echo_every,
    )
    torch.save(model.state_dict(), table_file)
    before, after = report.validation["forces"]
    print(f"Validation force RMS error: {before:.1f} -> {after:.1f} kJ/mol/nm")
    table = model.table.detach().cpu().numpy()[0]
    wall = (table[1] - table[0]) / (r[1] - r[0])
    print(f"Wall: u = {table[0]:.1f} kJ/mol at {start} nm, slope {wall:.0f} kJ/mol/nm")

    figure = Figure(figsize=(11.0, 4.2), layout="constrained")
    loss_axes, table_axes = figure.subplots(1, 2)
    window = 100  # updates; a batch's loss is mostly its frames' noise
    trend = np.convolve(report.loss, np.ones(window) / window, mode="valid")
    loss_axes.semilogy(np.arange(1, updates + 1), report.loss, linewidth=0.4, alpha=0.4)
    loss_axes.semilogy(np.arange(window, updates + 1), trend, color="black", label="mean of 100")
    loss_axes.set_xlabel("update")
    loss_axes.set_ylabel("batch loss (forces)")
    loss_axes.set_title(
        f"validation force RMS error {before:.1f} -> {after:.1f} kJ/mol/nm",
        fontsize=10,
    )
    loss_axes.legend()
    table_axes.plot(r, guess, ":", color="gray", label="start: -kT ln g(r)")
    table_axes.plot(r, table, "-", color="tab:green", label="force-matched table")
    table_axes.set_ylim(-3.0, 6.0)
    table_axes.set_xlabel("r (nm)")
    table_axes.set_ylabel("u(r) (kJ/mol)")
    table_axes.legend()
    figure.savefig(figure_file, dpi=150)
    print(f"Wrote {table_file} and {figure_file}")
    return model, report


if __name__ == "__main__":
    main()
