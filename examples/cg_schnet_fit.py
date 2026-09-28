"""Train a SchNet model of coarse-grained methanol by force matching at three densities.

Each ``data/methanol_cg_nvt*.npz`` file maps an all-atom NVT run of 343 methanol molecules
at 298.15 K to one bead per molecule: 1,000 frames 1 ps apart of the beads' positions and
forces, each bead's force the sum of its atoms' forces, with the beads' g(r). The densities
are 454.98, 473.54 (experiment) and 492.85 Da/nm^3. The model trains on the first 800 frames
of each file, validates on the rest and is saved for ``cg_schnet_nvt.py``, which runs it.
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from nabladyn import potential, training

DATA = Path(__file__).resolve().parent / "data"
FILES = ("methanol_cg_nvt_low.npz", "methanol_cg_nvt.npz", "methanol_cg_nvt_high.npz")


def main():
    dtype = torch.float32
    device = "cuda"
    cutoff = 1.0  # nm
    train_frames = 800  # of each file; the rest validate
    updates = 5000
    batch_size = 8  # frames
    learning_rate = 1e-3
    halve_every = 1000  # updates
    echo_every = 500  # updates
    weights_file = "cg_schnet.pt"
    figure_file = "cg_schnet_fit.png"

    sets = [np.load(DATA / name) for name in FILES]

    def frames(part):
        fields = ("positions", "cells", "forces")
        joined = {name: np.concatenate([data[name][part] for data in sets]) for name in fields}
        return training.Dataset(joined["positions"], joined["cells"], forces=joined["forces"])

    model = potential.schnet(
        atom_types=torch.as_tensor(sets[0]["atom_types"], device=device),
        num_types=1,
        cutoff=cutoff,
        trainable=True,
        dtype=dtype,
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
    torch.save(model.state_dict(), weights_file)
    before, after = report.validation["forces"]
    print(f"Validation force RMS error: {before:.1f} -> {after:.1f} kJ/mol/nm")

    figure = Figure(figsize=(6.5, 4.2), layout="constrained")
    axes = figure.subplots()
    window = 100  # updates; a batch's loss is mostly its frames' noise
    trend = np.convolve(report.loss, np.ones(window) / window, mode="valid")
    axes.semilogy(np.arange(1, updates + 1), report.loss, linewidth=0.4, alpha=0.4)
    axes.semilogy(np.arange(window, updates + 1), trend, color="black", label="mean of 100")
    axes.set_xlabel("update")
    axes.set_ylabel("batch loss (forces)")
    axes.set_title(
        f"validation force RMS error {before:.1f} -> {after:.1f} kJ/mol/nm",
        fontsize=10,
    )
    axes.legend()
    figure.savefig(figure_file, dpi=150)
    print(f"Wrote {weights_file} and {figure_file}")
    return model, report


if __name__ == "__main__":
    main()
