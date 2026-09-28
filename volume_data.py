"""Array loading and axis-aware volume selection for the 3D viewer.

Only NumPy is required. Spatial axes are explicit: singleton spatial axes are
never removed, and receive channels or time frames never become depth by
accident. CFL headers provide dimensions, not anatomical orientation or spacing.
"""

from collections.abc import Mapping
import math
from numbers import Integral
from pathlib import Path

import numpy as np


def _component_name(component):
    aliases = {"magnitude": "abs", "phase": "angle"}
    if not isinstance(component, str):
        raise ValueError("Component must be abs, real, imag, or angle.")
    name = aliases.get(component.lower(), component.lower())
    if name not in ("abs", "real", "imag", "angle"):
        raise ValueError("Component must be abs, real, imag, or angle.")
    return name


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


class VolumeDataset:
    """A numeric array with three explicitly identified spatial axes.

    ``spatial_axes`` lists the original array axes in output X/Y/Z order.
    ``spacing`` is also in that output order, in millimetres. If unspecified,
    spacing is one and coordinates are reported in voxels. Arrays with fewer
    than three axes acquire trailing singleton axes; no axes are squeezed.

    ``volume(indices={original_axis: index})`` selects nonspatial dimensions.
    Unspecified indices default to zero. Nonfinite values in the selected
    component are rejected rather than hidden or replaced.
    """

    def __init__(self, data, spatial_axes=(0, 1, 2), spacing=None, name="Volume"):
        array = np.asanyarray(data)
        if array.dtype.kind not in "biufc":
            raise ValueError("Volume data must be a numeric array, without objects.")
        if array.ndim == 0 or any(size == 0 for size in array.shape):
            raise ValueError("Volume data must have at least one nonempty axis.")
        self.raw_shape = tuple(array.shape)
        if array.ndim < 3:
            array = array.reshape(array.shape + (1,) * (3 - array.ndim))
        try:
            axes = tuple(spatial_axes)
        except TypeError as exc:
            raise ValueError("Spatial axes must contain three distinct axis numbers.") from exc
        if (len(axes) != 3 or any(not _integer(axis) for axis in axes)
                or len(set(axes)) != 3
                or any(axis < 0 or axis >= array.ndim for axis in axes)):
            raise ValueError(
                "Spatial axes must contain three distinct axis numbers in "
                "the range 0 through {}.".format(array.ndim - 1)
            )
        self.data = array
        self.spatial_axes = tuple(int(axis) for axis in axes)
        self.spatial_shape = tuple(array.shape[axis] for axis in self.spatial_axes)
        self._nonspatial_axes = tuple(
            axis for axis in range(array.ndim) if axis not in self.spatial_axes
        )
        self.extra_axes = [axis for axis in self._nonspatial_axes if array.shape[axis] > 1]
        self.extra_sizes = [array.shape[axis] for axis in self.extra_axes]
        self.name = str(name)
        self.units = "voxel" if spacing is None else "mm"
        if spacing is None:
            self.spacing = (1.0, 1.0, 1.0)
        else:
            try:
                values = tuple(float(value) for value in spacing)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Spacing must contain three positive finite numbers.") from exc
            if len(values) != 3 or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Spacing must contain three positive finite numbers.")
            self.spacing = values

    def volume(self, component="abs", indices=None):
        """Return a C-contiguous float32 XYZ volume for the selected frame."""
        component = _component_name(component)
        if indices is None:
            indices = {}
        if not isinstance(indices, Mapping):
            raise ValueError("Indices must map original nonspatial axis numbers to integers.")
        selection = [slice(None)] * self.data.ndim
        for axis in self._nonspatial_axes:
            selection[axis] = 0
        for axis, index in indices.items():
            if not _integer(axis) or axis not in self._nonspatial_axes:
                raise ValueError("Index axis {!r} is not a nonspatial axis.".format(axis))
            if not _integer(index) or index < 0 or index >= self.data.shape[axis]:
                raise ValueError(
                    "Index for axis {} must be an integer from 0 to {}.".format(
                        axis, self.data.shape[axis] - 1
                    )
                )
            selection[axis] = int(index)
        selected = self.data[tuple(selection)]
        retained_axes = sorted(self.spatial_axes)
        order = tuple(retained_axes.index(axis) for axis in self.spatial_axes)
        selected = np.transpose(selected, order)
        with np.errstate(over="ignore", invalid="ignore"):
            if component == "abs":
                # NumPy integer abs wraps for the most negative signed value.
                if selected.dtype.kind == "i":
                    selected = selected.astype(np.float64)
                result = np.abs(selected)
            elif component == "angle":
                result = np.angle(selected)
            elif component == "real":
                result = np.real(selected)
            else:
                result = np.imag(selected)
            result = np.ascontiguousarray(result, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError(
                "Selected {} volume contains nonfinite values or values outside "
                "the float32 range; inspect the source data.".format(component)
            )
        return result


def read_cfl(path):
    """Memory-map a BART CFL/HDR pair as little-endian complex64, Fortran order.

    Accept a base path or either filename. The header must contain exactly one
    non-comment dimensions line and the binary payload must match it exactly.
    """
    base = Path(path).expanduser()
    if base.suffix.lower() in (".cfl", ".hdr"):
        base = base.with_suffix("")
    header = Path(str(base) + ".hdr")
    payload = Path(str(base) + ".cfl")
    if not header.is_file() or not payload.is_file():
        raise FileNotFoundError("Expected both {} and {}.".format(header, payload))
    try:
        lines = [line.strip() for line in header.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    except UnicodeError as exc:
        raise ValueError("CFL header must be a text file containing dimensions.") from exc
    if len(lines) != 1:
        raise ValueError("CFL header must contain exactly one dimensions line.")
    try:
        dimensions = tuple(int(token) for token in lines[0].split())
    except ValueError as exc:
        raise ValueError("CFL dimensions must be positive integers.") from exc
    if not dimensions or any(size <= 0 for size in dimensions):
        raise ValueError("CFL dimensions must be positive integers.")
    dtype = np.dtype("<c8")
    expected_bytes = math.prod(dimensions) * dtype.itemsize
    actual_bytes = payload.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            "CFL payload size mismatch: header requires {} bytes, found {}.".format(
                expected_bytes, actual_bytes
            )
        )
    return np.memmap(payload, dtype=dtype, mode="r", shape=dimensions, order="F")


def load_dataset(path, spatial_axes=(0, 1, 2), spacing=None):
    """Load a CFL pair or a numeric .npy file without enabling pickle."""
    source = Path(path).expanduser()
    if source.suffix.lower() == ".npy":
        data = np.load(source, mmap_mode="r", allow_pickle=False)
        name = source.stem
    else:
        data = read_cfl(source)
        name = source.stem if source.suffix.lower() in (".cfl", ".hdr") else source.name
    return VolumeDataset(data, spatial_axes=spatial_axes, spacing=spacing, name=name)


def robust_clim(volume, component="abs"):
    """Suggest shared volume-wide contrast limits, using the 1st/99.5th percentiles.

    Phase always uses [-pi, pi]. Constant data receive a nonzero display range.
    Complex input and nonfinite data are rejected; pass a selected component.
    """
    component = _component_name(component)
    values = np.asarray(volume)
    if values.size == 0 or values.dtype.kind not in "biuf" or not np.isfinite(values).all():
        raise ValueError("Contrast limits require nonempty finite real values.")
    if component == "angle":
        return (-math.pi, math.pi)
    low, high = (float(value) for value in np.percentile(values, (1.0, 99.5)))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        # MRI reconstruction scales are arbitrary; a fixed minimum margin
        # would erase the contrast of small, nonzero constant arrays.
        margin = max(abs(low) * 0.01, np.finfo(float).tiny) if low else 0.5
        low, high = low - margin, high + margin
        if component == "abs" and low < 0:
            low = 0.0
    return (low, high)


def make_demo_dataset():
    """Create an asymmetric synthetic phantom; this is not a patient image."""
    coords = np.linspace(-1.0, 1.0, 96, dtype=np.float32)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij", sparse=True)
    radius = np.sqrt((x / 0.68) ** 2 + (y / 0.83) ** 2 + (z / 0.78) ** 2)
    inside = 1.0 / (1.0 + np.exp(np.clip((radius - 1.0) / 0.022, -60, 60)))
    core = 1.0 / (1.0 + np.exp(np.clip((radius - 0.74) / 0.035, -60, 60)))
    shell = np.exp(-((radius - 0.90) / 0.035) ** 2)
    signal = 0.38 * inside + 0.23 * core + 0.27 * shell
    # Off-centre dark cavity and differently sized bright landmarks reveal flips.
    cavity = np.exp(-(((x + 0.10) / 0.15) ** 2 + ((y + 0.03) / 0.23) ** 2
                       + ((z - 0.08) / 0.20) ** 2))
    landmark_a = np.exp(-(((x - 0.30) / 0.085) ** 2 + ((y - 0.12) / 0.10) ** 2
                          + ((z - 0.22) / 0.085) ** 2))
    landmark_b = np.exp(-(((x + 0.23) / 0.055) ** 2 + ((y + 0.30) / 0.08) ** 2
                          + ((z + 0.18) / 0.07) ** 2))
    signal = np.maximum(signal - 0.40 * cavity + 0.50 * landmark_a + 0.30 * landmark_b, 0)
    signal[radius > 1.12] = 0
    phase = 0.8 * x - 0.35 * z + 0.1 * y
    data = np.asarray(signal * np.exp(1j * phase), dtype=np.complex64)
    return VolumeDataset(data, spacing=(1.2, 1.2, 1.2), name="Synthetic demo phantom (not patient data)")
