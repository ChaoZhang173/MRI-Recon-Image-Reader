"""Public loading and axis-selection contracts across all CFL viewers."""

import contextlib
import io
import itertools
import os
from pathlib import Path
import struct
import tempfile
import unittest
import warnings

os.environ['MPLBACKEND'] = 'Agg'

import matplotlib.pyplot as plt
import numpy as np

import cflViewer
import cfl_viewer
import compareViewer
from volume_data import VolumeDataset, read_cfl


class ViewerConsistencyTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_asymmetric_xyz_planes_preserve_coordinates_in_every_frame(self):
        # Distinct axis lengths and coordinate digits expose permutations,
        # reflections, frame mixing, and accidental conversion of frames to Z.
        shape = (3, 5, 7, 1, 2, 1, 3)

        def voxel(x, y, z, echo, motion):
            return complex(x + 10*y + 100*z + 1000*echo + 10000*motion,
                           7 - x - 3*y - 9*z - 100*echo - 1000*motion)

        payload = bytearray()
        for motion, echo, z, y, x in itertools.product(
                range(3), range(2), range(7), range(5), range(3)):
            value = voxel(x, y, z, echo, motion)
            payload.extend(struct.pack('<ff', value.real, value.imag))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'asymmetric'
            Path(str(base) + '.hdr').write_text(
                '# Dimensions\n' + ' '.join(map(str, shape)) + '\n',
                encoding='utf-8')
            Path(str(base) + '.cfl').write_bytes(payload)
            data = read_cfl(base)
            stable = cflViewer.Viewer(data, title='xyz-stable')
            compare = compareViewer.CompareViewer(data, 2*data, title='xyz-compare')
            dataset = VolumeDataset(data)
            stable.part = compare.part = 'real'
            for echo, motion in itertools.product(range(2), range(3)):
                stable.extra_idx = compare.extra_idx = [echo, motion]
                volume = dataset.volume('real', {4: echo, 6: motion})
                for axis, direction in enumerate('xyz'):
                    remaining = [i for i in range(3) if i != axis]
                    stable.axis = compare.axis = direction
                    for index in range(shape[axis]):
                        with self.subTest(axis=direction, index=index,
                                          echo=echo, motion=motion):
                            expected = np.empty(
                                tuple(shape[i] for i in remaining), dtype=np.float32)
                            # Build the reference from coordinates, without
                            # reshaping or slicing the loaded volume.
                            for u, v in itertools.product(
                                    range(expected.shape[0]), range(expected.shape[1])):
                                xyz = [0, 0, 0]
                                xyz[axis] = index
                                xyz[remaining[0]] = u
                                xyz[remaining[1]] = v
                                expected[u, v] = voxel(*xyz, echo, motion).real
                            stable.slice_idx = compare.slice_idx = index
                            np.testing.assert_array_equal(
                                stable._current_image2d(), expected.T)
                            first, second, difference = compare._current_images2d()
                            np.testing.assert_array_equal(first, expected.T)
                            np.testing.assert_array_equal(second, 2*expected.T)
                            np.testing.assert_array_equal(difference, -expected.T)
                            np.testing.assert_array_equal(
                                np.take(volume, index, axis=axis), expected)

    def test_all_public_readers_match_independent_bart_bytes(self):
        shape = (2, 3, 1, 2, 1, 3, 1)
        expected = np.empty(shape, dtype=np.complex64)
        payload = bytearray()
        for motion in range(3):
            for echo in range(2):
                for y in range(3):
                    for x in range(2):
                        real = x + 10 * y + 100 * echo + 1000 * motion
                        imag = -real - 7
                        expected[x, y, 0, echo, 0, motion, 0] = complex(real, imag)
                        payload.extend(struct.pack('<ff', real, imag))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'motion.v1'
            Path(str(base) + '.hdr').write_text(
                '\n# Dimensions\n2 3 1 2 1 3 1\n', encoding='utf-8')
            Path(str(base) + '.cfl').write_bytes(payload)
            for reader in (cflViewer.read_cfl, compareViewer.read_cfl,
                           cfl_viewer.readcfl, read_cfl):
                with self.subTest(reader=reader):
                    actual = reader(Path(str(base) + '.hdr'))
                    np.testing.assert_array_equal(actual, expected)
            mapped = read_cfl(base)
            self.assertIsInstance(mapped, np.memmap)
            self.assertFalse(mapped.flags.writeable)
            dataset = VolumeDataset(mapped)
            np.testing.assert_array_equal(
                dataset.volume('real', {3: 1, 5: 2}),
                expected[:, :, :, 1, 0, 2, 0].real,
            )

    def test_common_slider_dimensions_match_legacy_and_3d_selection(self):
        shape = (3, 4, 1, 2, 1, 3)
        data = np.arange(np.prod(shape)).reshape(shape) + 1j
        legacy = cfl_viewer.AdvancedCFLViewer.__new__(cfl_viewer.AdvancedCFLViewer)
        processed = legacy.process_data_dimensions(data)
        for helper in (cflViewer.squeeze_keep_extras, compareViewer.squeeze_keep_extras):
            actual, extras, merge = helper(data)
            self.assertEqual(extras, [2, 3])
            self.assertEqual(merge, 1)
            np.testing.assert_array_equal(actual, processed)
        dataset = VolumeDataset(data)
        np.testing.assert_array_equal(
            processed[:, :, :, 1, 2].real,
            dataset.volume('real', {3: 1, 5: 2}),
        )

    def test_legacy_overflow_preserves_all_frames(self):
        shape = (2, 3, 1, 2, 3, 2, 2)
        data = np.arange(np.prod(shape)).reshape(shape)
        legacy = cfl_viewer.AdvancedCFLViewer.__new__(cfl_viewer.AdvancedCFLViewer)
        with contextlib.redirect_stdout(io.StringIO()):
            processed = legacy.process_data_dimensions(data)
        self.assertEqual(processed.shape, (2, 3, 1, 2, 12))
        np.testing.assert_array_equal(processed[:, :, :, 1, 11], data[:, :, :, 1, 2, 1, 1])
        self.assertEqual(processed.size, data.size)

    def test_stable_and_compare_preserve_singleton_z_and_switch_slice_ranges(self):
        shape = (3, 4, 1, 2, 1, 3)
        data = (np.arange(np.prod(shape)).reshape(shape) + 1j).astype(np.complex64)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            stable = cflViewer.Viewer(data, title='singleton-stable')
            compare = compareViewer.CompareViewer(data, data, title='singleton-compare')
            for viewer in (stable, compare):
                self.assertEqual((viewer.X, viewer.Y, viewer.Z), (3, 4, 1))
                self.assertEqual(viewer.extra_sizes, [2, 3])
                self.assertFalse(viewer.slider_slice.active)
                viewer.extra_idx = [1, 2]
                viewer.part = 'real'
                expected = data[:, :, 0, 1, 0, 2].real.T
                if viewer is stable:
                    np.testing.assert_array_equal(viewer._current_image2d(), expected)
                else:
                    first, second, difference = viewer._current_images2d()
                    np.testing.assert_array_equal(first, expected)
                    np.testing.assert_array_equal(second, expected)
                    np.testing.assert_array_equal(difference, np.zeros_like(expected))
                viewer._on_radio_axis('x')
                self.assertTrue(viewer.slider_slice.active)
                self.assertEqual(viewer.slider_slice.valmax, 3)
                viewer._on_radio_axis('z')
                self.assertFalse(viewer.slider_slice.active)
                self.assertEqual(viewer.slice_idx, 0)

    def test_stable_and_compare_pad_one_and_two_dimensional_inputs(self):
        for shape in ((3,), (3, 4)):
            with self.subTest(shape=shape), warnings.catch_warnings():
                warnings.simplefilter('error', UserWarning)
                data = np.arange(np.prod(shape)).reshape(shape).astype(np.complex64)
                stable = cflViewer.Viewer(data, title='padded-stable')
                compare = compareViewer.CompareViewer(data, data, title='padded-compare')
                expected = shape + (1,) * (3 - len(shape))
                self.assertEqual(stable.data.shape, expected)
                self.assertEqual(compare.data1.shape, expected)
                plt.close('all')

    def test_all_stable_extra_sliders_fit_in_figure(self):
        data = np.ones((2, 3, 1, 2, 2, 2, 3), dtype=np.complex64)
        stable = cflViewer.Viewer(data, title='extras-stable')
        compare = compareViewer.CompareViewer(data, data, title='extras-compare')
        for viewer in (stable, compare):
            self.assertEqual(viewer.extra_sizes, [2, 2, 6])
            self.assertEqual(viewer.extra_sliders[-1].label.get_text(), 'D6+')
            for slider in viewer.extra_sliders:
                self.assertGreaterEqual(slider.ax.get_position().y0, 0)


if __name__ == '__main__':
    unittest.main()
