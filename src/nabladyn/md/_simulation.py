"""The simulation: it owns a running frame, its neighbor list and its CUDA Graph."""

import math
import os
from numbers import Integral, Real
from typing import TextIO

import torch
from torch import Tensor

from .. import quantity, space
from .._compile import _compile
from ..neighbors import NeighborList
from ..observables import kinetic_energy, pressure, temperature
from ..potential import Potential
from ._output import FRAME, Echo, SampleWriter
from ._state import State, _Frame

_STATUS_INTERVAL = 1000  # steps between reads of the neighbor list's status
# The sampled panel, in kJ/mol, kJ/mol, kJ/mol, K and kJ/mol/nm^3.
FIELDS = ("potential_energy", "kinetic_energy", "total_energy", "temperature", "pressure")


def _positive(name: str, value: Real) -> float:
    """Return ``value`` as a float; raise ``ValueError`` unless it is finite and positive."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _count(name: str, value: Integral) -> int:
    """Return ``value`` as an int; raise ``ValueError`` unless it is a nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _capture_graph(program, *args) -> torch.cuda.CUDAGraph:
    """Capture ``program(*args)`` in a CUDA Graph that owns its cuBLAS workspace.

    Every capture runs on torch's one capture stream, and cuBLAS keeps one workspace per
    stream. Allocated during the first capture, it lives in that graph's private memory and
    is freed with that graph, while later graphs that reused it would go on reading and
    writing it: an illegal memory access once the memory is released. Clearing the
    workspaces before and after each capture gives every graph its own. Syncs with the host.
    """
    torch._C._cuda_clearCublasWorkspaces()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program(*args)
    torch._C._cuda_clearCublasWorkspaces()
    return graph


def _check_model(model: Potential) -> None:
    """Raise ``TypeError`` unless the model is a ``Potential``."""
    if not isinstance(model, Potential):
        raise TypeError("model must be a Potential")


def _check_frame(
    model: Potential,
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    cell: Tensor,
) -> None:
    """Raise ``ValueError`` unless each tensor has its shape and the model's dtype and device."""
    num_atoms, dtype, device = model.num_atoms, model.dtype, model.device
    expected = (
        ("positions", positions, (num_atoms, 3)),
        ("velocities", velocities, (num_atoms, 3)),
        ("masses", masses, (num_atoms,)),
        ("cell", cell, (3,)),
    )
    for name, value, shape in expected:
        if (
            not isinstance(value, Tensor)
            or value.shape != shape
            or value.dtype != dtype
            or value.device != device
        ):
            raise ValueError(
                f"{name} must be a {list(shape)} tensor of {dtype} on {device} to match the model",
            )


def _check_images(images: Tensor, num_atoms: int, device: torch.device) -> None:
    """Raise ``ValueError`` unless the images are an int64 [N, 3] tensor on the device."""
    if (
        not isinstance(images, Tensor)
        or images.shape != (num_atoms, 3)
        or images.dtype != torch.int64
        or images.device != device
    ):
        raise ValueError(f"images must be a [{num_atoms}, 3] int64 tensor on {device}")


def _check_velocities(velocities: Tensor) -> None:
    """Raise ``ValueError`` unless every velocity is finite."""
    if not bool(torch.isfinite(velocities).all()):
        raise ValueError("velocities must be finite")


def _check_masses(masses: Tensor) -> None:
    """Raise ``ValueError`` unless every mass is finite and positive."""
    if not bool((torch.isfinite(masses) & (masses > 0)).all()):
        raise ValueError("masses must be finite and positive")


def _check_sampling(sample_every: Integral | None, echo, sample_file) -> None:
    """Raise ``ValueError`` unless the sampling, echo and sample file settings fit together."""
    if sample_every is not None and _count("sample_every", sample_every) == 0:
        raise ValueError("sample_every must be positive")
    if echo is not False and echo is not True and not hasattr(echo, "write"):
        raise ValueError("echo must be True, False or a text stream")
    if (echo is not False or sample_file is not None) and sample_every is None:
        raise ValueError("echo and sample_file need sample_every")


class Simulation:
    """Molecular dynamics of a potential on the GPU; build it with ``nve``, ``nvt`` or ``npt``.

    It owns its frame, a neighbor list at the model's cutoff, with an inner list at each
    smaller cutoff the model reads, and a CUDA Graph of one step, which ``run`` replays. A
    second graph, captured on first use, evaluates the panel of ``FIELDS``: potential,
    kinetic and total energy in kJ/mol, the temperature over $`3N - 3`$ degrees of freedom in
    K, and the pressure in kJ/mol/nm^3. The step reads the model's parameter storage, so a
    parameter changed in place reaches the next step.
    """

    def __init__(
        self,
        model: Potential,
        positions: Tensor,
        velocities: Tensor,
        masses: Tensor,
        cell: Tensor,
        *,
        ensemble,
        timestep: Real,
        skin: Real,
        capacity_multiplier: Real,
        images: Tensor | None,
        step: Integral,
    ):
        _check_model(model)
        num_atoms, device = model.num_atoms, model.device
        _check_frame(model, positions, velocities, masses, cell)
        if images is None:
            images = torch.zeros((num_atoms, 3), dtype=torch.int64, device=device)
        else:
            _check_images(images, num_atoms, device)
        timestep = _positive("timestep", timestep)
        self._step = _count("step", step)
        velocities, masses = velocities.detach(), masses.detach().clone()
        _check_velocities(velocities)
        _check_masses(masses)
        self._neighbors = NeighborList(
            model.cutoff,
            skin=skin,
            capacity_multiplier=capacity_multiplier,
            inner_cutoffs=model._list_cutoffs[:-1],
        )
        self._neighbors.allocate(positions, cell)  # checks the positions and the cell
        cell = cell.detach().clone()
        positions, crossings = space.wrap(positions.detach(), cell)
        # The center-of-mass velocity is a constant of motion; at zero it leaves 3N - 3
        # degrees of freedom for the temperature and the thermostat.
        column = masses[:, None]
        velocities = velocities - (column * velocities).sum(dim=0) / masses.sum()
        evaluate = quantity.energy_force_virial(model._energy_wrapped)
        # No autograd graph: a trained model's first frame would hold it for the whole run.
        with torch.no_grad():
            energy, forces, virial = evaluate(positions, cell, self._neighbors.tensors())
        self._frame = _Frame(
            positions=positions,
            velocities=velocities,
            forces=forces,
            energy=energy,
            virial=virial,
            cell=cell,
            images=images + crossings,
        )
        self._model, self._masses, self._timestep = model, masses, timestep
        self._degrees_of_freedom = 3 * num_atoms - 3
        self._ensemble = ensemble(timestep, masses, self._frame)
        self._fields = FIELDS + self._ensemble.fields
        self._failure: str | None = None
        self._programs, self._graph = self._capture()
        self._sampler = None  # (program, graph, row), captured on first use

    def run(
        self,
        steps: Integral,
        *,
        sample_every: Integral | None = None,
        echo: bool | TextIO = False,
        sample_file: str | os.PathLike | None = None,
    ) -> dict[str, Tensor] | None:
        """Advance ``steps`` steps.

        With ``sample_every``, returns the panel at each step that is a multiple of it, as
        host tensors [S] under the names of ``FIELDS`` and the ensemble's fields, and
        ``step`` [S], int64; otherwise None. ``echo`` (True for stdout, or a text stream)
        prints a header and one row per sample. ``sample_file`` names an HDF5 file,
        overwritten, holding each sample's frame and panel (see ``read_sample_file``). Both
        need ``sample_every``.

        Syncs with the host: reads the neighbor list's status every 1,000 steps, before each
        echoed or written sample, and at the end.

        Raises ``ValueError`` for settings out of range, ``RuntimeError`` after a failure.
        A neighbor list error raises ``ValueError``, discards the call's samples and fails
        the simulation; the file keeps the samples written before it.
        """
        steps = _count("steps", steps)
        _check_sampling(sample_every, echo, sample_file)
        self._require()
        start, due, rows = self._step, [], None
        if sample_every is not None:
            first = (start // sample_every + 1) * sample_every
            due = list(range(first, start + steps + 1, sample_every))
            rows = self._frame.energy.new_empty((len(due), len(self._fields)))
            row = self._sample_graph()[2]
        printer = None if echo is False else Echo(echo, self._fields)
        writer = None
        if sample_file is not None:
            writer = SampleWriter(
                sample_file,
                masses=self._masses.cpu().numpy(),
                fields=self._fields,
                attributes={
                    **self._ensemble.settings,
                    "timestep": self._timestep,
                    "cutoff": self._model.cutoff,
                },
            )
        try:
            for done in range(1, steps + 1):
                self._graph.replay()
                step = start + done
                if due and step % sample_every == 0:
                    self._sampler[1].replay()
                    rows[step // sample_every - due[0] // sample_every].copy_(row)
                    if printer is not None or writer is not None:
                        self._check(step)  # no frame of a failed run is written
                        values, time = row.tolist(), step * self._timestep
                        if printer is not None:
                            printer.write(step, time, values)
                        if writer is not None:
                            frame = {
                                name: getattr(self._frame, name).cpu().numpy() for name in FRAME
                            }
                            writer.write(step, time, frame, values)
                if done % _STATUS_INTERVAL == 0 or done == steps:
                    self._check(step)
        finally:
            if writer is not None:
                writer.close()
        self._step += steps
        if sample_every is None:
            return None
        rows = rows.cpu()
        samples = {"step": torch.tensor(due, dtype=torch.int64)}
        samples.update({name: rows[:, index] for index, name in enumerate(self._fields)})
        return samples

    @property
    def model(self) -> Potential:
        """The model the simulation runs."""
        return self._model

    @property
    def settings(self) -> dict:
        """The ensemble's settings, such as its name and temperature; a copy."""
        return dict(self._ensemble.settings)

    @property
    def masses(self) -> Tensor:
        """The masses [N] in Da; a copy."""
        return self._masses.clone()

    def sample(self) -> dict[str, float]:
        """Return the current frame's panel under the names of ``FIELDS`` and the ensemble's.

        Syncs with the host. Raises ``RuntimeError`` after a failure.
        """
        self._require()
        _, graph, row = self._sample_graph()
        graph.replay()
        return dict(zip(self._fields, row.tolist()))

    def state(self) -> State:
        """Return a copy of the current frame as a ``State``.

        Raises ``RuntimeError`` after a failure.
        """
        self._require()
        frame = self._frame
        return State(
            frame.positions.clone(),
            frame.velocities.clone(),
            frame.forces.clone(),
            frame.cell.clone(),
            frame.images.clone(),
            self._step,
            **self._ensemble.state(),
        )

    def _capture(self):
        """Compile the step's halves, warm them up and capture one step in a CUDA Graph.

        The ensemble's halves update the frame in place: ``first_half(frame)`` before the
        neighbor update and the force at the new positions, ``second_half(frame)`` after.
        The warm-up advances the frame and the ensemble's tensors; both are restored after.
        """
        frame, ensemble, neighbors = self._frame, self._ensemble, self._neighbors
        pairs = neighbors.tensors()
        energy_fn = self._model._energy_wrapped
        if ensemble.needs_virial:
            evaluate = quantity.energy_force_virial(energy_fn)
        else:
            evaluate = quantity.energy_and_force(energy_fn)

        def second_half(frame: _Frame, pairs) -> None:
            energy, forces, *virial = evaluate(frame.positions, frame.cell, pairs)
            frame.energy.copy_(energy)
            frame.forces.copy_(forces)
            if virial:
                frame.virial.copy_(virial[0])
            ensemble.second_half(frame)

        programs = [_compile(ensemble.first_half), _compile(second_half)]
        if ensemble.closing is not None:
            programs.append(_compile(ensemble.closing))

        def step() -> None:
            programs[0](frame)
            neighbors._update_wrapped(frame.positions, frame.cell)
            programs[1](frame, pairs)
            for program in programs[2:]:
                program(frame)

        resident = (*frame, *ensemble.tensors())
        entry = [tensor.clone() for tensor in resident]
        current = torch.cuda.current_stream(frame.positions.device)
        stream = torch.cuda.Stream(frame.positions.device)
        stream.wait_stream(current)
        with torch.no_grad():
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            current.wait_stream(stream)
            self._check(self._step + 3)  # catches a first frame that fails in the warm-up
            graph = _capture_graph(step)
        for tensor, saved in zip(resident, entry):
            tensor.copy_(saved)
        return programs, graph

    def _sample_graph(self):
        """Return the panel program, its CUDA Graph and its output row, captured on first use.

        The program reads the current frame and writes only the row.
        """
        if self._sampler is not None:
            return self._sampler
        frame, masses, degrees = self._frame, self._masses, self._degrees_of_freedom
        ensemble, pairs = self._ensemble, self._neighbors.tensors()
        energy_and_virial = quantity.energy_virial(self._model._energy_wrapped)
        row = frame.energy.new_empty(len(self._fields))

        def panel(frame: _Frame, pairs, row: Tensor) -> None:
            energy, virial = energy_and_virial(frame.positions, frame.cell, pairs)
            kinetic = kinetic_energy(frame.velocities, masses)
            values = (
                energy,
                kinetic,
                energy + kinetic,
                temperature(kinetic, degrees),
                pressure(kinetic, virial, frame.cell),
                *ensemble.panel(frame, energy, kinetic),
            )
            row.copy_(torch.stack(values))

        program = _compile(panel)
        current = torch.cuda.current_stream(row.device)
        stream = torch.cuda.Stream(row.device)
        stream.wait_stream(current)
        with torch.no_grad():
            with torch.cuda.stream(stream):
                for _ in range(2):
                    program(frame, pairs, row)
            current.wait_stream(stream)
            graph = _capture_graph(program, frame, pairs, row)
        self._sampler = (program, graph, row)
        return self._sampler

    def _check(self, step: int) -> None:
        """Raise ``ValueError``, and fail the simulation, unless the neighbor list is sound.

        Syncs with the host.
        """
        try:
            self._neighbors._check_status()
        except ValueError as error:
            self._failure = f"by step {step}: {error}"
            raise ValueError(f"the simulation failed {self._failure}") from error

    def _require(self) -> None:
        """Raise ``RuntimeError`` if the simulation has failed."""
        if self._failure is not None:
            raise RuntimeError(
                f"the simulation failed {self._failure}; start a new one from an earlier "
                "state()",
            )


__all__ = ("FIELDS", "Simulation")
