"""Exercise the reported legacy viewer through its actual CFL input path."""

import contextlib
import io
import os
from pathlib import Path
import struct
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from cfl_reader import readcfl
from cfl_viewer import AdvancedCFLViewer


class LegacyViewerInputTests(unittest.TestCase):
    def test_complex_frames_and_xyz_slices_reach_the_display_unchanged(self):
        # A separately encoded BART fixture: X changes fastest; frames remain
        # outside XYZ, with singleton axes between the two frame dimensions.
        shape = (4, 5, 3, 1, 2, 1, 3)
        expected = np.empty(shape, dtype=np.complex64)
        payload = bytearray()
        for motion in range(3):
            for echo in range(2):
                for z in range(3):
                    for y in range(5):
                        for x in range(4):
                            real = x + 10*y + 100*z + 1000*echo + 10000*motion
                            imag = 7 - real
                            expected[x, y, z, 0, echo, 0, motion] = complex(real, imag)
                            payload.extend(struct.pack("<ff", real, imag))
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "scan"
            Path(str(base) + ".hdr").write_text(
                "# Dimensions\n" + " ".join(map(str, shape)) + "\n"
            )
            Path(str(base) + ".cfl").write_bytes(payload)
            with contextlib.redirect_stdout(io.StringIO()):
                viewer = AdvancedCFLViewer(readcfl(base))
                self.addCleanup(plt.close, viewer.fig)
                self.assertEqual(viewer.processed_data.shape, (4, 5, 3, 2, 3))
                viewer.slider_dim1.set_val(1)
                viewer.slider_dim2.set_val(2)
                volume = expected[:, :, :, 0, 1, 0, 2]
                for axis_number, direction in enumerate("xyz"):
                    viewer.on_slice_direction_change(direction)
                    viewer.slider_slice.set_val(1)
                    plane = np.take(volume, 1, axis=axis_number)
                    np.testing.assert_array_equal(viewer.get_current_slice_data(), plane)
                    for mode, component in (
                        ("magnitude", np.abs), ("real", np.real),
                        ("imaginary", np.imag), ("phase", np.angle),
                    ):
                        with self.subTest(direction=direction, mode=mode):
                            viewer.on_mode_change(mode)
                            np.testing.assert_array_equal(
                                viewer.im.get_array(), component(plane)
                            )


if __name__ == "__main__":
    unittest.main()
