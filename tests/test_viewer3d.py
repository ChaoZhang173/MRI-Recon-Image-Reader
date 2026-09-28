"""Opt-in native GUI integration tests; needs an available graphics display.

MRI_VIEWER_GUI_TESTS=1 python -m unittest discover -s tests -p test_viewer3d.py -v
Set MRI_VIEWER_SCREENSHOT to save the synthetic demo window as a PNG.
"""
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("MRI_VIEWER_GUI_TESTS") == "1", "Native GUI tests are opt-in")
class ViewerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        from viewer3d import ViewerWindow
        from volume_data import make_demo_dataset
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.window = ViewerWindow(make_demo_dataset())
        cls.window.show()
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.app.processEvents()

    def test_demo_pixels_and_controls(self):
        from pathlib import Path
        from volume_data import make_demo_dataset
        w = self.window
        w.set_dataset(make_demo_dataset())
        self.app.processEvents()
        pixels = w.plotter.screenshot(return_img=True)
        self.assertGreater(np.std(pixels), 10)
        height, width = pixels.shape[:2]
        for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
            panel = pixels[y*height//2:(y+1)*height//2, x*width//2:(x+1)*width//2]
            # Exclude text and borders: the centre must contain rendered data.
            centre = panel[panel.shape[0]//4:3*panel.shape[0]//4, panel.shape[1]//4:3*panel.shape[1]//4]
            self.assertGreater(np.std(centre), 8)
        w.opacity.setValue(65)
        self.assertAlmostEqual(w.scene._opacity, 0.65)
        w.threshold.setValue(25)
        w.cut_enabled.setChecked(True)
        self.assertEqual(w.scene.volume_actor.GetMapper().GetNumberOfClippingPlanes(), 1)
        w.cut_enabled.setChecked(False)
        w.cmap.setCurrentText("magma")
        self.assertEqual(w.scene._cmap, "magma")
        w.cmap.setCurrentText("gray")
        w.opacity.setValue(35)
        w.threshold.setValue(10)
        screenshot = os.environ.get("MRI_VIEWER_SCREENSHOT")
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            self.app.processEvents()
            # QWidget.grab() omits the native VTK surface on macOS.
            w.plotter.screenshot(screenshot)
            self.assertTrue(Path(screenshot).is_file())

    def test_rejected_frame_rolls_back_selection(self):
        from volume_data import VolumeDataset
        w = self.window
        data = np.ones((8, 9, 10, 3), dtype=np.complex64)
        data[..., 1] = 2
        data[..., 2] = np.nan
        w.set_dataset(VolumeDataset(data))
        w.extra_controls[3].setValue(1)
        self.assertTrue(np.all(w.scene.volume == 2))
        errors = []
        original = w._report_error
        w._report_error = lambda title, exc: errors.append(str(exc))
        try:
            w.extra_controls[3].setValue(2)
        finally:
            w._report_error = original
        self.assertEqual(len(errors), 1)
        self.assertEqual(w.extra_controls[3].value(), 1)
        self.assertTrue(np.all(w.scene.volume == 2))

    def test_tiny_amplitudes_and_slice_selection(self):
        from volume_data import VolumeDataset
        w = self.window
        data = np.arange(8*9*10, dtype=np.float32).reshape(8, 9, 10) * 1e-15
        w.set_dataset(VolumeDataset(data, spacing=(0.4, 0.7, 1.8)))
        self.assertGreater(w.maximum.value(), w.minimum.value())
        self.assertLess(w.maximum.value(), 1e-10)
        w.slice_sliders[2].setValue(8)
        self.assertEqual(w.scene.slice_indices[2], 8)
        np.testing.assert_array_equal(
            w.scene.slice_meshes[2].point_data["intensity"], data[:, :, 8].ravel(order="F")
        )

    def test_geometry_and_singleton_depth(self):
        from volume_data import VolumeDataset
        w = self.window
        w.set_dataset(VolumeDataset(np.ones((8, 9, 1, 2), dtype=np.float32)))
        self.assertEqual(w.scene.volume.shape, (8, 9, 1))
        self.assertIsNone(w.scene.volume_actor)
        self.assertFalse(w.slice_sliders[2].isEnabled())
        self.assertFalse(w.known_spacing.isChecked())
        self.assertEqual(w.extra_controls[3].maximum(), 1)
        w.known_spacing.setChecked(True)
        for control, value in zip(w.spacing_controls, (0.5, 0.7, 2.1)):
            control.setValue(value)
        w.apply_geometry()
        self.assertEqual(w.dataset.units, "mm")
        np.testing.assert_allclose(w.scene.grid.spacing, (0.5, 0.7, 2.1))


if __name__ == "__main__":
    unittest.main()
