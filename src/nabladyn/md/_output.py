"""Sample output: the echo and HDF5 sample files."""

import os
import sys

import numpy as np

SCHEMA, SCHEMA_VERSION = "nabladyn.samples", 1
UNITS = {
    "step": "1",
    "time": "ps",
    "masses": "Da",
    "positions": "nm",
    "velocities": "nm/ps",
    "forces": "kJ/mol/nm",
    "cell": "nm",
    "images": "1",
    "potential_energy": "kJ/mol",
    "kinetic_energy": "kJ/mol",
    "total_energy": "kJ/mol",
    "temperature": "K",
    "pressure": "kJ/mol/nm^3",
    "volume": "nm^3",
    "conserved_energy": "kJ/mol",
}
# The frame arrays each sample stores; all but images are in the run's dtype.
FRAME = ("positions", "velocities", "forces", "cell", "images")


class Echo:
    """A text echo: a header, then one row per sample of step, time and panel, with units."""

    def __init__(self, stream, fields: tuple[str, ...]):
        self._stream = sys.stdout if stream is True else stream
        header = ["step", "time[ps]", *(f"{name}[{UNITS[name]}]" for name in fields)]
        self._stream.write(" ".join(header) + "\n")

    def write(self, step: int, time: float, values: list[float]) -> None:
        """Write one sample's row."""
        columns = [str(step), f"{time:.10g}", *(f"{value:.10g}" for value in values)]
        self._stream.write(" ".join(columns) + "\n")


class SampleWriter:
    """An HDF5 sample file, created or overwritten when opened.

    Each sample appends one frame and one panel row; every dataset carries its unit.
    """

    def __init__(
        self,
        path,
        *,
        masses: np.ndarray,
        fields: tuple[str, ...],
        attributes: dict,
    ):
        import h5py  # optional; only a run that writes a file needs it

        self._file = h5py.File(os.fspath(path), "w")
        self._file.attrs.update({"schema": SCHEMA, "schema_version": SCHEMA_VERSION, **attributes})
        self._write("masses", masses)
        num_atoms, dtype = masses.shape[0], masses.dtype
        shapes = {
            "step": ((), np.int64),
            "time": ((), np.float64),
            "positions": ((num_atoms, 3), dtype),
            "velocities": ((num_atoms, 3), dtype),
            "forces": ((num_atoms, 3), dtype),
            "cell": ((3,), dtype),
            "images": ((num_atoms, 3), np.int64),
            **{name: ((), dtype) for name in fields},
        }
        for name, (shape, kind) in shapes.items():
            dataset = self._file.create_dataset(
                name,
                (0, *shape),
                maxshape=(None, *shape),
                dtype=kind,
                chunks=(64, *shape),
            )
            dataset.attrs["unit"] = UNITS[name]
        self._names = tuple(shapes)
        self._count = 0

    def write(self, step: int, time: float, frame: dict, values: list[float]) -> None:
        """Append one sample: its step, time, frame arrays and panel values."""
        row = {
            "step": step,
            "time": time,
            **frame,
            **dict(zip(self._names[len(FRAME) + 2:], values)),
        }
        for name in self._names:
            dataset = self._file[name]
            dataset.resize(self._count + 1, axis=0)
            dataset[self._count] = row[name]
        self._count += 1

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    def _write(self, name: str, values: np.ndarray) -> None:
        """Write one dataset of ``values`` with its unit."""
        self._file.create_dataset(name, data=values).attrs["unit"] = UNITS[name]


def read_sample_file(path: str | os.PathLike) -> tuple[dict[str, np.ndarray], dict]:
    """Read a sample file's samples and attributes.

    Samples are NumPy arrays under their dataset names: ``step``, ``time``, the frame arrays
    ``positions``, ``velocities``, ``forces``, ``cell`` and ``images``, ``masses`` and the
    panel fields. Attributes are the ensemble and its settings, the timestep in ps and the
    cutoff in nm. The file holds no thermostat or barostat state, so it is not a restart;
    continue a run from ``Simulation.state()``.

    Raises ``ValueError`` for a file that is not a NablaDyn sample file.
    """
    import h5py  # optional; only reading a sample file needs it

    with h5py.File(os.fspath(path), "r") as file:
        if file.attrs.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a NablaDyn sample file")
        samples = {name: file[name][()] for name in file}
        attributes = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in file.attrs.items()
        }
    return samples, attributes


__all__ = ("read_sample_file",)
