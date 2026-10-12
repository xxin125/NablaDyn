"""Train an angular SchNet on top of a pair-table prior by force matching, coarse-grained methanol.

The prior is the force-matched pair table that ``cg_table_fit.py`` saves (run it first), held
as it is: its 1.0 nm reach carries the weak attraction between 0.6 and 1.0 nm, which the
angular SchNet at 0.6 nm cannot see. The model is their sum, E = E_table + E_network, so
fitting it fits the network to the forces the table misses. Data and protocol are those of
``cg_angular_schnet_fit.py``: the three ``data/methanol_cg_nvt*.npz`` files, 343 one-bead
methanol molecules at 298.15 K and 454.98, 473.54 and 492.85 Da/nm^3, the first 800 frames
of each training and the rest validating. The network is saved for
``cg_prior_angular_schnet_nvt.py``, which runs the sum.
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
    table_cutoff = 1.0  # nm, as in cg_table_fit.py
    cutoff = 0.6  # nm, the network's
    orders = 5  # the network's highest order of angle features
    train_frames = 800  # of each file; the rest validate
    updates = 5000
    batch_size = 8  # frames
    learning_rate = 1e-3
    halve_every = 1000  # updates
    echo_every = 500  # updates
    table_file = "cg_table.pt"  # from cg_table_fit.py
    weights_file = "cg_prior_angular_schnet.pt"
    figure_file = "cg_prior_angular_schnet_fit.png"

    sets = [np.load(DATA / name) for name in FILES]

    def frames(part):
        fields = ("positions", "cells", "forces")
        joined = {name: np.concatenate([data[name][part] for data in sets]) for name in fields}
        return training.Dataset(joined["positions"], joined["cells"], forces=joined["forces"])

    types = torch.as_tensor(sets[0]["atom_types"], device=device)
    saved = torch.load(table_file, map_location=device)
    prior = potential.tabulated(
        atom_types=types,
        table=saved["table"].to(dtype),
        start=saved["start"].to(dtype),
        cutoff=table_cutoff,
    )
    network = potential.angular_schnet(
        atom_types=types,
        num_types=1,
        cutoff=cutoff,
        orders=orders,
        trainable=True,
        dtype=dtype,
    )
    model = potential.sum(prior, network)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
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
    torch.save(network.state_dict(), weights_file)
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
        f"table + angular SchNet: validation force RMS error {before:.1f} -> {after:.1f} "
        "kJ/mol/nm",
        fontsize=10,
    )
    axes.legend()
    figure.savefig(figure_file, dpi=150)
    print(f"Wrote {weights_file} and {figure_file}")
    return model, report


if __name__ == "__main__":
    main()
