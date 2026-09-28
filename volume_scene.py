"""Rendering for the 3D viewer, kept independent of the Qt controls.

Coordinates are array X/Y/Z axes. No anatomical orientation is inferred.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction, vtkPlane
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper


class VolumeScene:
    """One volume and three orthogonal, voxel-aligned slices in a 2x2 plotter."""

    PANELS = ((0, 1), (1, 0), (1, 1))
    AXES = "XYZ"
    BACKGROUND = "#101923"
    FOREGROUND = "#edf2f7"

    def __init__(self, plotter):
        self.plotter = plotter
        self.volume = None
        self.grid = None
        self.volume_actor = None
        self.slice_indices = [0, 0, 0]
        self.slice_meshes = [None, None, None]
        self._slice_actors = [None, None, None]
        self._flat_actor = None
        self._mapper = None
        self._property = None
        self._spacing = (1.0, 1.0, 1.0)
        self._units = "voxel"
        self._component = "abs"
        self._clim = (0.0, 1.0)
        self._opacity = 0.25
        self._threshold = 0.15
        self._cmap = "gray"
        self._blend = "composite"
        self._cutaway = (False, 0, 0.5, False)
        self._clip_plane = None
        for row in range(2):
            for col in range(2):
                plotter.subplot(row, col)
                plotter.set_background(self.BACKGROUND)
        plotter.subplot(0, 0)
        plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z", color=self.FOREGROUND)

    @staticmethod
    def _check_clim(clim):
        low, high = (float(value) for value in clim)
        if not np.isfinite([low, high]).all() or high < low:
            raise ValueError("Intensity limits must be finite and ordered.")
        if high == low:
            high = low + max(abs(low) * 1e-6, 1e-6)
        return low, high

    def _text(self, message, name, position="upper_left"):
        self.plotter.add_text(
            message,
            position=position,
            name=name,
            color=self.FOREGROUND,
            font_size=10,
            shadow=False,
            render=False,
        )

    def set_volume(
        self,
        volume3d,
        spacing=(1.0, 1.0, 1.0),
        units="voxel",
        component="abs",
        clim=(0.0, 1.0),
        reset_camera=True,
    ):
        """Set an X/Y/Z array without rescaling its scalar values.

        When a new frame has the same shape, its current slice indices and, with
        ``reset_camera=False``, all camera positions are retained.
        """
        values = np.asarray(volume3d)
        if values.ndim != 3 or any(size == 0 for size in values.shape):
            raise ValueError("The rendered volume must have three nonempty axes.")
        if np.iscomplexobj(values):
            raise ValueError("Choose a real-valued complex component before rendering.")
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("The rendered volume must contain finite values.")
        spacing = tuple(float(value) for value in spacing)
        if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError("Spacing must contain three finite positive values.")
        limits = self._check_clim(clim)
        first_volume = self.volume is None
        shape_changed = first_volume or self.volume.shape != values.shape
        if shape_changed:
            self.slice_indices = [size // 2 for size in values.shape]

        self.volume = values
        self._spacing = spacing
        self._units = str(units)
        self._component = str(component)
        self._clim = limits
        self.grid = pv.ImageData(dimensions=values.shape, spacing=spacing)
        self.grid.point_data["intensity"] = values.ravel(order="F")

        self.plotter.subplot(0, 0)
        for name in ("volume", "flat-volume", "volume-outline"):
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        self.volume_actor = None
        self._flat_actor = None
        self._mapper = None
        self._property = None
        spatial_dimension = sum(size > 1 for size in values.shape)
        if spatial_dimension == 3:
            self._mapper = vtkSmartVolumeMapper()
            self._mapper.SetInputData(self.grid)
            self._property = vtkVolumeProperty()
            self._property.SetInterpolationTypeToLinear()
            self._property.SetScalarOpacityUnitDistance(min(spacing))
            self._property.ShadeOn()
            self._property.SetAmbient(0.25)
            self._property.SetDiffuse(0.8)
            self._property.SetSpecular(0.15)
            self.volume_actor = vtkVolume()
            self.volume_actor.SetMapper(self._mapper)
            self.volume_actor.SetProperty(self._property)
            self.plotter.add_actor(
                self.volume_actor, name="volume", reset_camera=False, render=False
            )
            title = "3D volume"
        elif spatial_dimension == 2:
            self._flat_actor = self.plotter.add_mesh(
                self.grid,
                scalars="intensity",
                cmap=self._cmap,
                clim=self._clim,
                lighting=False,
                show_scalar_bar=False,
                name="flat-volume",
                reset_camera=False,
                render=False,
            )
            title = "2D data · no 3D thickness"
        else:
            title = "No 3D volume\nFewer than two spatial dimensions"
        self._text(title, "volume-title")
        self._text(
            f"Data axes X / Y / Z · {self._units} · {self._component}",
            "volume-caption",
            position="lower_left",
        )
        if spatial_dimension >= 2:
            self.plotter.add_mesh(
                self.grid.outline(),
                color="#617184",
                line_width=1,
                name="volume-outline",
                reset_camera=False,
                render=False,
            )
        for axis in range(3):
            self._update_slice(axis)
        self._update_appearance()
        self._update_cutaway()
        if reset_camera or first_volume:
            self.reset_views()
        else:
            for renderer in self.plotter.renderers:
                renderer.reset_camera_clipping_range()
            self.plotter.render()

    def _update_slice(self, axis):
        self.plotter.subplot(*self.PANELS[axis])
        self.plotter.remove_actor(f"slice-{axis}", reset_camera=False, render=False)
        self._slice_actors[axis] = None
        self.slice_meshes[axis] = None
        index = self.slice_indices[axis]
        shape = self.volume.shape
        plane_axes = [value for value in range(3) if value != axis]
        horizontal, vertical = ((1, 2), (0, 2), (0, 1))[axis]
        title = f"{self.AXES[axis]} slice · index {index} / {shape[axis] - 1}"
        self._text(title, f"slice-title-{axis}")
        self._text(
            f"{self.AXES[horizontal]}–{self.AXES[vertical]} plane · data axes",
            f"slice-caption-{axis}",
            position="lower_left",
        )
        self.plotter.remove_actor(f"slice-absent-{axis}", reset_camera=False, render=False)
        if any(shape[value] == 1 for value in plane_axes):
            self._text(
                title + "\nNo 2D plane (one axis has size 1)",
                f"slice-title-{axis}",
            )
            return
        extent = [0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1]
        extent[2 * axis : 2 * axis + 2] = [index, index]
        mesh = self.grid.extract_subset(extent)
        self.slice_meshes[axis] = mesh
        self._slice_actors[axis] = self.plotter.add_mesh(
            mesh,
            scalars="intensity",
            cmap=self._cmap,
            clim=self._clim,
            lighting=False,
            show_scalar_bar=False,
            interpolate_before_map=True,
            name=f"slice-{axis}",
            reset_camera=False,
            render=False,
        )
        self.plotter.renderer.reset_camera_clipping_range()

    def set_slice(self, axis, index):
        if self.volume is None:
            return
        if axis not in (0, 1, 2):
            raise ValueError("Slice axis must be 0, 1, or 2.")
        index = int(index)
        if not 0 <= index < self.volume.shape[axis]:
            raise ValueError("Slice index is outside the volume.")
        if self.slice_indices[axis] == index:
            return
        self.slice_indices[axis] = index
        self._update_slice(axis)
        self.plotter.render()

    def set_appearance(self, clim, opacity, threshold, cmap, blend="composite"):
        """Update transfer functions; threshold is relative to the given limits."""
        limits = self._check_clim(clim)
        if not 0 <= opacity <= 1 or not 0 <= threshold <= 1:
            raise ValueError("Opacity and threshold must lie between 0 and 1.")
        if blend not in ("composite", "maximum"):
            raise ValueError("Blend mode must be composite or maximum.")
        # Validate before replacing the current appearance.
        pv.LookupTable(cmap=cmap)
        self._clim = limits
        self._opacity = float(opacity)
        self._threshold = float(threshold)
        self._cmap = cmap
        self._blend = blend
        self._update_appearance()
        self.plotter.render()

    def _update_appearance(self):
        if self.grid is None:
            return
        lut = pv.LookupTable(cmap=self._cmap, scalar_range=self._clim)
        for actor in [*self._slice_actors, self._flat_actor]:
            if actor is not None:
                actor.mapper.lookup_table = lut
                actor.mapper.scalar_range = self._clim
        if self._property is None:
            return
        low, high = self._clim
        color = vtkColorTransferFunction()
        for value, rgba in zip(np.linspace(low, high, lut.n_values), lut.values):
            color.AddRGBPoint(float(value), *(rgba[:3] / 255.0))
        alpha = vtkPiecewiseFunction()
        cutoff = low + self._threshold * (high - low)
        alpha.AddPoint(low, 0.0)
        alpha.AddPoint(cutoff, 0.0)
        alpha.AddPoint(high, self._opacity if cutoff < high else 0.0)
        self._property.SetColor(color)
        self._property.SetScalarOpacity(alpha)
        if self._blend == "maximum":
            self._mapper.SetBlendModeToMaximumIntensity()
        else:
            self._mapper.SetBlendModeToComposite()

    def set_cutaway(self, enabled, axis=0, fraction=0.5, flip=False):
        if axis not in (0, 1, 2) or not 0 <= fraction <= 1:
            raise ValueError("Cutaway needs a valid axis and fraction from 0 to 1.")
        self._cutaway = bool(enabled), int(axis), float(fraction), bool(flip)
        self._update_cutaway()
        self.plotter.render()

    def _update_cutaway(self):
        self._clip_plane = None
        if self._mapper is None:
            return
        self._mapper.RemoveAllClippingPlanes()
        enabled, axis, fraction, flip = self._cutaway
        if enabled:
            bounds = self.grid.bounds
            origin = list(self.grid.center)
            origin[axis] = bounds[2 * axis] + fraction * (
                bounds[2 * axis + 1] - bounds[2 * axis]
            )
            normal = [0.0, 0.0, 0.0]
            normal[axis] = -1.0 if flip else 1.0
            self._clip_plane = vtkPlane()
            self._clip_plane.SetOrigin(origin)
            self._clip_plane.SetNormal(normal)
            self._mapper.AddClippingPlane(self._clip_plane)

    def _face_axis(self, axis, mesh):
        center = np.asarray(mesh.center)
        direction = np.eye(3)[axis]
        if axis == 1:
            direction *= -1
        up = (0, 1, 0) if axis == 2 else (0, 0, 1)
        position = center + direction * max(self.grid.length * 2, 1.0)
        self.plotter.camera_position = (tuple(position), tuple(center), up)
        self.plotter.enable_parallel_projection()
        self.plotter.reset_camera(bounds=mesh.bounds, render=False)

    def reset_views(self):
        if self.grid is None:
            return
        self.plotter.subplot(0, 0)
        if sum(size > 1 for size in self.volume.shape) == 2:
            axis = next(index for index, size in enumerate(self.volume.shape) if size == 1)
            self._face_axis(axis, self.grid)
        else:
            self.plotter.disable_parallel_projection()
            self.plotter.view_isometric(render=False)
            self.plotter.reset_camera(bounds=self.grid.bounds, render=False)
        for axis, mesh in enumerate(self.slice_meshes):
            if mesh is not None:
                self.plotter.subplot(*self.PANELS[axis])
                self._face_axis(axis, mesh)
        self.plotter.subplot(0, 0)
        self.plotter.render()
