"""Fitting a potential's parameters to labeled frames.

Frames are prepared once with their pair stacks; compiled programs give a chunk's loss
and gradient, or its channels; ``fit`` runs the updates.
"""

import dataclasses
import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import Tensor, nn

from .. import space
from .._compile import _compile
from ..neighbors import NeighborList
from ..neighbors._neighbor_list import _OVERFLOW
from ..potential import Potential
from . import _loss
from ._dataset import Dataset


def _chunks(frames: Tensor, size: int):
    """Yield (chunk, number of real frames) in chunks of ``size``.

    A short chunk is padded by repeating its first frame.
    """
    for part in frames.split(size):
        yield torch.cat([part, part[:1].expand(size - len(part))]), len(part)


def _batch_gradient(
    program,
    parameters,
    prepared,
    frames: Tensor,
    size: int,
    weights,
    scales,
):
    """Return one batch's gradient, loss and terms, summed over chunks of ``size`` frames.

    Real frames weigh 1/B and padding frames 0, so the chunk sums equal the batch's exactly.
    """
    positions, cells, stacks, labels = prepared
    total = None
    for chunk, real in _chunks(frames, size):
        chunk = chunk.to(positions.device)
        coefficients = torch.zeros(size, dtype=positions.dtype, device=positions.device)
        coefficients[:real] = 1.0 / len(frames)
        gradient, (loss, terms) = program(
            parameters,
            positions[chunk],
            cells[chunk],
            tuple(stack[chunk] for stack in stacks),
            {name: value[chunk] for name, value in labels.items()},
            coefficients,
            weights,
            scales,
        )
        if total is None:
            total = (gradient, loss, terms)
        else:
            total = (
                {name: total[0][name] + value for name, value in gradient.items()},
                total[1] + loss,
                {name: total[2][name] + value for name, value in terms.items()},
            )
    return total


def _prepared(model: Potential, data: Dataset, channels: tuple[str, ...]):
    """Return (positions, cells, pair stacks, labels) on the model's device and in its dtype.

    The pair stacks are a tuple, one per cutoff of ``model._list_cutoffs``.
    """

    def move(value: Tensor) -> Tensor:
        return value.to(device=model.device, dtype=model.dtype)

    positions, cells = move(data.positions), move(data.cells)
    labels = {name: move(data.labels[name]) for name in channels}
    stacks = tuple(pair_stack(positions, cells, cutoff) for cutoff in model._list_cutoffs)
    return positions, cells, stacks, labels


def _errors(program, prepared, channels: tuple[str, ...], size: int) -> dict[str, float]:
    """Return each channel's RMS error over a prepared set, in chunks of ``size`` frames.

    Syncs with the host.
    """
    positions, cells, stacks, labels = prepared
    squares = dict.fromkeys(channels, 0.0)
    with torch.no_grad():
        for chunk, real in _chunks(torch.arange(len(positions)), size):
            chunk = chunk.to(positions.device)
            values = program(
                positions[chunk],
                cells[chunk],
                tuple(stack[chunk] for stack in stacks),
            )
            for name in channels:
                error = values[name][:real] - labels[name][chunk[:real]]
                squares[name] = squares[name] + error.square().sum()
    return {name: math.sqrt(float(squares[name]) / labels[name].numel()) for name in channels}


def _check_model(model: Potential) -> None:
    """Raise ``TypeError`` unless the model is a Potential."""
    if not isinstance(model, Potential):
        raise TypeError("model must be a Potential")


def _check_weights(weights: dict[str, Real]) -> None:
    """Raise ``ValueError`` unless some weight is given and all are finite and positive."""
    if not weights:
        raise ValueError("give at least one channel's weight: energy, forces or virial")
    for name, value in weights.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"the {name} weight must be a finite positive number")


def _check_counts(
    updates: Integral,
    batch_size: Integral,
    seed: Integral,
    echo_every: Integral | None,
    frames: int,
) -> None:
    """Raise ``ValueError`` unless the counts are nonnegative integers in their ranges."""
    counts = (
        ("updates", updates),
        ("batch_size", batch_size),
        ("seed", seed),
        ("echo_every", 1 if echo_every is None else echo_every),
    )
    for name, value in counts:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if updates == 0 or echo_every == 0 or not 0 < batch_size <= frames:
        raise ValueError(
            "updates and echo_every must be positive, and batch_size between 1 and the "
            "training set's size",
        )


def _check_trained(trained: dict[str, Tensor]) -> None:
    """Raise ``ValueError`` unless the model trains at least one parameter."""
    if not trained:
        raise ValueError(
            "the model has no trainable parameter; set trainable when building the potential",
        )


def _check_sets(model: Potential, sets: dict[str, Dataset], channels: tuple[str, ...]) -> None:
    """Raise ``ValueError`` unless every set has the model's atoms and the channels' labels."""
    for label, data in sets.items():
        if data.positions.shape[1] != model.num_atoms:
            raise ValueError(
                f"the {label} frames hold {data.positions.shape[1]} atoms, the model "
                f"{model.num_atoms}",
            )
        for name in channels:
            if name not in data.labels:
                raise ValueError(f"the {label} frames have no {name} labels")


@dataclasses.dataclass(frozen=True)
class Report:
    """The result of a fit.

    ``loss`` [updates]: each update's batch loss, before its step. ``scales``: the channel
    scales from the training labels. ``training`` and ``validation`` (None without a
    validation set): each weighted channel's RMS error (before, after) the fit, for the
    energy per frame in kJ/mol, force components in kJ/mol/nm and virial components in kJ/mol.
    """

    loss: np.ndarray
    scales: dict[str, float]
    training: dict[str, tuple[float, float]]
    validation: dict[str, tuple[float, float]] | None


def width(count: int) -> int:
    """Return ``count`` rounded up to a multiple of 1/8 of its leading power of two.

    The width grows at most 12.5%, so similar datasets share one width and one compiled
    program; a vmapped gradient recompiles for each width.
    """
    step = 1 << max(0, count.bit_length() - 4)
    return -(-count // step) * step


def pair_stack(positions: Tensor, cells: Tensor, cutoff: float) -> Tensor:
    """Return the pairs within ``cutoff`` of frames [F, N, 3] in cells [F, 3], int32 [F, 2, C].

    Row f holds frame f's pairs, then self-pairs (k mod N, k mod N) up to the common width C,
    which holds the most pairs of any frame. A zero-skin list counts each frame's pairs, then
    one sized for the densest frame builds every row. Syncs with the host four times, not per
    frame, and twice more each time a frame has over twice the pairs the count was sized for.

    Raises ``ValueError`` for a cell length that is not finite and above twice the cutoff, or
    a non-finite coordinate.
    """
    frames, atoms = positions.shape[:2]
    wrapped = space.wrap(positions.detach(), cells.detach()[:, None])[0].contiguous()
    cells = cells.detach().contiguous()
    totals = torch.empty(frames, dtype=torch.int32, device=positions.device)
    densest = 0
    while True:
        counter = NeighborList(cutoff, capacity_multiplier=2.0)
        counter.allocate(wrapped[densest], cells[densest])
        for frame in range(frames):
            counter._rebuild_wrapped(wrapped[frame], cells[frame])
            totals[frame : frame + 1].copy_(counter._buffers.pair_total)
        status = counter._buffers.status[0]
        flags, densest = torch.stack([status, totals.argmax().to(status.dtype)]).tolist()
        if flags & ~_OVERFLOW:
            counter._check_status()
        if not flags & _OVERFLOW:
            break
        # An overflowing list builds nothing more, so the first overflowing frame holds the
        # largest count (argmax takes the first of equals); the recount is sized at it, with
        # at least twice the room that overflowed.
    neighbors = NeighborList(cutoff, capacity_multiplier=1.0)
    neighbors.allocate(wrapped[densest], cells[densest])
    capacity = neighbors.pair_capacity
    columns = width(capacity)
    fill = torch.arange(columns, dtype=torch.int32, device=positions.device) % atoms
    stack = fill.expand(frames, 2, columns).contiguous()
    pairs = neighbors.tensors().index
    for frame in range(frames):
        neighbors._rebuild_wrapped(wrapped[frame], cells[frame])
        stack[frame, :, :capacity].copy_(pairs)
    neighbors._check_status()
    return stack


def gradient_program(model: nn.Module, channels: tuple[str, ...]):
    """Return the compiled loss of a chunk of m frames and its gradient in the trained parameters.

    The program takes (parameters, positions [m, N, 3], cells [m, 3], stacks, labels,
    coefficients [m], weights, scales) and returns (gradient, (loss, terms)). ``parameters``
    maps trained parameter names to values; ``stacks`` holds the pair indices [m, 2, C_k],
    one per cutoff of the model's ``_list_cutoffs``. Coefficients, weights and scales are
    tensors, so new values do not recompile.
    """

    def loss_and_gradient(
        parameters,
        positions,
        cells,
        stacks,
        labels,
        coefficients,
        weights,
        scales,
    ):
        def loss(trained):
            predicted = _loss.predict(model, trained, positions, cells, stacks, channels)
            return _loss.loss(predicted, labels, weights, scales, coefficients)

        return torch.func.grad_and_value(loss, has_aux=True)(parameters)

    return _compile(loss_and_gradient)


def values_program(model: nn.Module, channels: tuple[str, ...]):
    """Return the compiled channels of a chunk of frames at the model's own parameters.

    The program takes (positions [m, N, 3], cells [m, 3], stacks), the stacks as for
    ``gradient_program``, and returns the channels by name.
    """

    def values(positions: Tensor, cells: Tensor, stacks: tuple[Tensor, ...]) -> dict[str, Tensor]:
        return _loss.predict(model, {}, positions, cells, stacks, channels)

    return _compile(values)


def batches(frames: int, batch_size: int, updates: int, seed: int) -> list[Tensor]:
    """Return the frame indices of each update.

    Each epoch is a new shuffle from ``seed``, cut into batches of ``batch_size``; its last
    batch holds the remainder.
    """
    generator = torch.Generator().manual_seed(seed)
    result = []
    while len(result) < updates:
        result.extend(torch.randperm(frames, generator=generator).split(batch_size))
    return result[:updates]


def held_entries(model: Potential, trained: dict[str, Tensor]):
    """Return (parameter, held-entry mask, copy of its values) per partly trained parameter."""
    masks = model._trained_masks()
    return [
        (value, ~masks[name], value.detach().clone())
        for name, value in trained.items()
        if name in masks
    ]


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    trained: dict[str, Tensor],
    gradient: dict[str, Tensor],
    held,
) -> None:
    """Write ``gradient`` into the trained parameters' ``.grad`` and step the optimizer.

    Held entries get zero gradient and are restored after the step, since weight decay moves
    them even at zero gradient.
    """
    optimizer.zero_grad()
    for name, value in trained.items():
        value.grad = gradient[name]
    for value, mask, values in held:
        value.grad = value.grad.masked_fill(mask, 0)
    optimizer.step()
    with torch.no_grad():
        for value, mask, values in held:
            value.copy_(torch.where(mask, values, value))


def fit(
    model: Potential,
    train: Dataset,
    optimizer: torch.optim.Optimizer,
    *,
    updates: Integral,
    batch_size: Integral,
    energy: Real | None = None,
    forces: Real | None = None,
    virial: Real | None = None,
    validation: Dataset | None = None,
    seed: Integral = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    echo_every: Integral | None = None,
) -> Report:
    r"""Fit the model's trained parameters to the training labels and return a ``Report``.

    Trained parameters are those that require gradients. A channel enters the loss when its
    weight w is given (``energy``, ``forces``, ``virial``), as
    $`w \cdot \mathrm{mean}[((X - X^*) / s)^2]`$ with the scale s from the training
    labels. Each of the ``updates`` takes the next batch
    of ``batch_size`` frames (each epoch a new shuffle from ``seed``, its last batch the
    remainder), writes the gradient of the batch's mean loss into ``.grad`` and calls
    ``optimizer.step()``. Rows the model does not train keep their values. Parameters change
    in place; another call continues from them and the optimizer. ``scheduler``, built on the
    optimizer, steps after each update; one that needs a metric, such as
    ``ReduceLROnPlateau``, is not supported. The report gives the RMS errors before and after
    on the training frames and on ``validation``, which is not trained on.

    With ``echo_every`` = k, prints a header, then every k updates the update number, the
    mean loss and the mean unweighted terms over those k updates.

    Syncs with the host.

    Raises ``TypeError`` for a model that is not a Potential, and ``ValueError`` for weights
    or counts out of range, a model that trains no parameter, or frames with another atom
    count or without the weighted labels.
    """
    _check_model(model)
    weights = {
        name: value
        for name, value in (("energy", energy), ("forces", forces), ("virial", virial))
        if value is not None
    }
    _check_weights(weights)
    _check_counts(updates, batch_size, seed, echo_every, len(train))
    trained = {name: value for name, value in model.named_parameters() if value.requires_grad}
    _check_trained(trained)
    held = held_entries(model, trained)
    channels = tuple(name for name in _loss.CHANNELS if name in weights)
    sets = {"training": train}
    if validation is not None:
        sets["validation"] = validation
    _check_sets(model, sets, channels)
    prepared = {label: _prepared(model, data, channels) for label, data in sets.items()}
    scales = _loss.scales(**prepared["training"][3])

    def tensors(values):
        return {
            name: torch.tensor(value, dtype=model.dtype, device=model.device)
            for name, value in values.items()
        }

    values = values_program(model, channels)
    before = {
        label: _errors(values, data, channels, batch_size) for label, data in prepared.items()
    }
    program = gradient_program(model, channels)
    weights, channel_scales = tensors(weights), tensors(scales)
    losses = torch.empty(updates, dtype=torch.float64, device=model.device)
    terms = torch.empty((updates, len(channels)), dtype=torch.float64, device=model.device)
    if echo_every is not None:
        print("update loss " + " ".join(channels), flush=True)
    for update, frames in enumerate(batches(len(train), batch_size, updates, seed)):
        gradient, loss, loss_terms = _batch_gradient(
            program,
            {name: value.detach() for name, value in trained.items()},
            prepared["training"],
            frames,
            batch_size,
            weights,
            channel_scales,
        )
        optimizer_step(optimizer, trained, gradient, held)
        if scheduler is not None:
            scheduler.step()
        losses[update] = loss
        terms[update] = torch.stack([loss_terms[name] for name in channels])
        if echo_every is not None and (update + 1) % echo_every == 0:
            interval = slice(update + 1 - echo_every, update + 1)
            means = torch.cat([losses[interval].mean().reshape(1), terms[interval].mean(dim=0)])
            print(
                f"{update + 1} " + " ".join(f"{value:.6g}" for value in means.tolist()),
                flush=True,
            )
    after = {
        label: _errors(values, data, channels, batch_size) for label, data in prepared.items()
    }
    errors = {
        label: {name: (before[label][name], after[label][name]) for name in channels}
        for label in prepared
    }
    return Report(
        loss=losses.cpu().numpy(),
        scales=scales,
        training=errors["training"],
        validation=errors.get("validation"),
    )


__all__ = ("Report", "fit")
