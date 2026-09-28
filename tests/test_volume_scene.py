"""Geometry and transfer-function tests; these do not validate rendered pixels.

Rendering is suppressed before scene construction and cleanup because a Cocoa
VTK offscreen window may have no OpenGL context. Qt-backed visual checks remain
separate from these tests.
"""

import unittest

import numpy as np
import pyvista as pv

from volume_scene import VolumeScene


class VolumeSceneTests(unittest.TestCase):
    def setUp(self):
        self.plotter = pv.Plotter(shape=(2, 2), off_screen=True)
        self.plotter.render = lambda: None
        self.addCleanup(self._close_plotter)
        self.scene = VolumeScene(self.plotter)
        self.shape = (7, 9, 11)
        x, y, z = np.indices(self.shape)
        self.values = (100 * x + 10 * y + z).astype(np.float32)
        self.spacing = (2.0, 3.0, 4.0)
        self.scene.set_volume(
            self.values, spacing=self.spacing, clim=(0, float(self.values.max()))
        )

    def _close_plotter(self):
        self.plotter.render = lambda: None
        self.plotter.close()

    def _camera_states(self):
        return [
            (
                tuple(renderer.camera.position),
                tuple(renderer.camera.focal_point),
                tuple(renderer.camera.up),
                renderer.camera.parallel_projection,
                renderer.camera.parallel_scale,
            )
            for renderer in self.plotter.renderers
        ]

    def test_point_values_keep_xyz_order_float_intensity_and_spacing(self):
        scalars = self.scene.grid.point_data["intensity"]
        self.assertEqual(scalars.dtype, np.float32)
        np.testing.assert_array_equal(scalars, self.values.ravel(order="F"))
        self.assertGreater(float(scalars.max()), 255)
        np.testing.assert_allclose(self.scene.grid.bounds, (0, 12, 0, 24, 0, 40))
        # A known voxel ties its scalar to its physical location, not just shape.
        voxel = (2, 3, 4)
        flat_index = np.ravel_multi_index(voxel, self.shape, order="F")
        self.assertEqual(scalars[flat_index], 234)
        np.testing.assert_allclose(self.scene.grid.points[flat_index], (4, 9, 16))

    def test_exact_boundary_and_middle_slices_in_all_axes(self):
        for axis in range(3):
            for index in (0, self.shape[axis] // 2, self.shape[axis] - 1):
                with self.subTest(axis=axis, index=index):
                    self.scene.set_slice(axis, index)
                    mesh = self.scene.slice_meshes[axis]
                    actual = mesh.point_data["intensity"].reshape(
                        mesh.dimensions, order="F"
                    )
                    np.testing.assert_array_equal(
                        actual, np.take(self.values, [index], axis=axis)
                    )
                    expected_position = index * self.spacing[axis]
                    self.assertAlmostEqual(mesh.bounds[2 * axis], expected_position)
                    self.assertAlmostEqual(mesh.bounds[2 * axis + 1], expected_position)

    def test_parallel_cameras_match_displayed_positive_axis_labels(self):
        horizontal_axes = (1, 0, 0)
        vertical_axes = (2, 2, 1)
        for axis, panel in enumerate(self.scene.PANELS):
            with self.subTest(axis=axis):
                self.plotter.subplot(*panel)
                camera = self.plotter.camera
                outward = np.asarray(camera.position) - np.asarray(camera.focal_point)
                outward /= np.linalg.norm(outward)
                expected = np.eye(3)[axis] * (-1 if axis == 1 else 1)
                np.testing.assert_allclose(outward, expected)
                up = np.asarray(camera.up)
                screen_right = np.cross(-outward, up)
                np.testing.assert_allclose(screen_right, np.eye(3)[horizontal_axes[axis]])
                np.testing.assert_allclose(up, np.eye(3)[vertical_axes[axis]])
                self.assertTrue(camera.parallel_projection)

    def test_transfer_functions_use_original_intensities_without_mutation(self):
        original = self.scene.grid.point_data["intensity"].copy()
        self.scene.set_appearance(
            (200, 500), opacity=0.6, threshold=0.5, cmap="viridis"
        )
        opacity = self.scene.volume_actor.GetProperty().GetScalarOpacity()
        self.assertEqual(opacity.GetValue(349), 0)
        self.assertAlmostEqual(opacity.GetValue(425), 0.3)
        self.assertAlmostEqual(opacity.GetValue(500), 0.6)
        np.testing.assert_array_equal(self.scene.grid.point_data["intensity"], original)
        self.scene.set_appearance(
            (200, 500), opacity=0.6, threshold=1.0, cmap="gray", blend="maximum"
        )
        opacity = self.scene.volume_actor.GetProperty().GetScalarOpacity()
        self.assertEqual(opacity.GetValue(500), 0)
        self.assertEqual(self.scene.volume_actor.GetMapper().GetBlendMode(), 1)

    def test_cutaway_affects_only_volume_and_can_flip_or_clear(self):
        before = [mesh.point_data["intensity"].copy() for mesh in self.scene.slice_meshes]
        self.scene.set_cutaway(True, axis=1, fraction=0.25)
        mapper = self.scene.volume_actor.GetMapper()
        self.assertEqual(mapper.GetNumberOfClippingPlanes(), 1)
        plane = mapper.GetClippingPlanes().GetItem(0)
        np.testing.assert_allclose(plane.GetOrigin(), (6, 6, 20))
        np.testing.assert_array_equal(plane.GetNormal(), (0, 1, 0))
        self.scene.set_cutaway(True, axis=1, fraction=0.25, flip=True)
        plane = mapper.GetClippingPlanes().GetItem(0)
        np.testing.assert_array_equal(plane.GetNormal(), (0, -1, 0))
        for original, mesh in zip(before, self.scene.slice_meshes):
            np.testing.assert_array_equal(original, mesh.point_data["intensity"])
        self.scene.set_cutaway(False)
        self.assertEqual(mapper.GetNumberOfClippingPlanes(), 0)

    def test_frame_change_retains_camera_and_slice_selection(self):
        self.scene.set_slice(0, 1)
        self.scene.set_slice(1, 7)
        self.scene.set_slice(2, 9)
        self.plotter.subplot(0, 0)
        self.plotter.camera.azimuth += 17
        self.plotter.camera.zoom(1.4)
        cameras = self._camera_states()
        indices = self.scene.slice_indices.copy()
        self.scene.set_volume(
            self.values + 1000,
            spacing=self.spacing,
            clim=(1000, 2000),
            reset_camera=False,
        )
        self.assertEqual(self._camera_states(), cameras)
        self.assertEqual(self.scene.slice_indices, indices)
        for axis, mesh in enumerate(self.scene.slice_meshes):
            actual = mesh.point_data["intensity"].reshape(mesh.dimensions, order="F")
            np.testing.assert_array_equal(
                actual, np.take(self.values + 1000, [indices[axis]], axis=axis)
            )

    def test_singletons_are_planes_or_absent_not_invented_volume_thickness(self):
        for singleton in range(3):
            with self.subTest(singleton=singleton):
                shape = [7, 8, 9]
                shape[singleton] = 1
                values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                self.scene.set_volume(values, spacing=self.spacing, clim=(0, values.max()))
                self.assertIsNone(self.scene.volume_actor)
                self.assertEqual(
                    [mesh is not None for mesh in self.scene.slice_meshes],
                    [axis == singleton for axis in range(3)],
                )
                self.assertEqual(self.scene.grid.bounds[2 * singleton], 0)
                self.assertEqual(self.scene.grid.bounds[2 * singleton + 1], 0)
        self.scene.set_volume(np.arange(9, dtype=np.float32).reshape(1, 1, 9), clim=(0, 8))
        self.assertIsNone(self.scene.volume_actor)
        self.assertTrue(all(mesh is None for mesh in self.scene.slice_meshes))

    def test_constant_data_and_rejected_nonfinite_input(self):
        self.scene.set_volume(np.ones((2, 2, 2), dtype=np.float32), clim=(1, 1))
        opacity = self.scene.volume_actor.GetProperty().GetScalarOpacity()
        low, high = opacity.GetRange()
        self.assertGreater(high, low)
        self.assertTrue(np.isfinite([opacity.GetValue(low), opacity.GetValue(high)]).all())
        for value in (np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                self.scene.set_volume(np.full((2, 2, 2), value, dtype=np.float32))
        np.testing.assert_array_equal(self.scene.volume, 1)


if __name__ == "__main__":
    unittest.main()
