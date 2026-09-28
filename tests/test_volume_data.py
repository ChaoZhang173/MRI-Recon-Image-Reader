"""Meaningful data-integrity checks independent of graphics or Qt."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from volume_data import VolumeDataset, load_dataset, make_demo_dataset, read_cfl, robust_clim


class VolumeSelectionTests(unittest.TestCase):
    def test_singleton_spatial_axis_does_not_become_time(self):
        raw = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 1, 3)
        dataset = VolumeDataset(raw)
        self.assertEqual(dataset.raw_shape, (4, 5, 1, 3))
        self.assertEqual(dataset.spatial_shape, (4, 5, 1))
        self.assertEqual(dataset.extra_axes, [3])
        self.assertEqual(dataset.extra_sizes, [3])
        actual = dataset.volume(indices={3: 2})
        self.assertEqual(actual.shape, (4, 5, 1))
        np.testing.assert_array_equal(actual, raw[:, :, :, 2])

    def test_permutation_and_original_extra_axis_numbers(self):
        raw = np.arange(2 * 3 * 4 * 5 * 1 * 2).reshape(2, 3, 4, 5, 1, 2)
        dataset = VolumeDataset(raw, spatial_axes=(3, 0, 2), spacing=(0.5, 1, 2))
        actual = dataset.volume("real", {1: 2, 5: 1})
        expected = raw[:, 2, :, :, 0, 1].transpose(2, 0, 1)
        self.assertEqual(dataset.spatial_shape, (5, 2, 4))
        self.assertEqual(dataset.extra_axes, [1, 5])
        self.assertEqual(dataset.spacing, (0.5, 1.0, 2.0))
        self.assertEqual(dataset.units, "mm")
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_2d_is_padded_without_invented_depth(self):
        dataset = VolumeDataset(np.ones((4, 6)))
        self.assertEqual(dataset.raw_shape, (4, 6))
        self.assertEqual(dataset.spatial_shape, (4, 6, 1))
        self.assertEqual(dataset.volume().shape, (4, 6, 1))
        self.assertEqual(dataset.units, "voxel")
        self.assertEqual(dataset.spacing, (1.0, 1.0, 1.0))

    def test_components_and_default_indices(self):
        raw = np.full((2, 3, 4, 2), 3 + 4j, dtype=np.complex64)
        raw[..., 1] = -2 + 0j
        dataset = VolumeDataset(raw)
        np.testing.assert_array_equal(dataset.volume("abs"), 5)
        np.testing.assert_array_equal(dataset.volume("magnitude"), 5)
        np.testing.assert_array_equal(dataset.volume("real"), 3)
        np.testing.assert_array_equal(dataset.volume("imag"), 4)
        np.testing.assert_allclose(dataset.volume("phase"), math.atan2(4, 3), rtol=1e-6)
        np.testing.assert_array_equal(dataset.volume("real", {3: 1}), -2)
        with self.assertRaisesRegex(ValueError, "Component"):
            dataset.volume("unknown")

    def test_bad_spacing_and_axes(self):
        raw = np.ones((2, 3, 4))
        for spacing in [(1, 0, 2), (1, -1, 2), (1, np.nan, 2), (1, np.inf, 2), (1, 2), "bad", 1]:
            with self.subTest(spacing=spacing), self.assertRaisesRegex(ValueError, "Spacing"):
                VolumeDataset(raw, spacing=spacing)
        for axes in [(0, 0, 2), (0, 1, 3), (-1, 1, 2), (0, 1), (0, 1, 2.0), (False, 1, 2), None]:
            with self.subTest(axes=axes), self.assertRaisesRegex(ValueError, "Spatial axes"):
                VolumeDataset(raw, spatial_axes=axes)

    def test_bad_indices_and_nonnumeric_data(self):
        dataset = VolumeDataset(np.ones((2, 3, 4, 5)))
        for indices in [{0: 0}, {9: 0}, {3: -1}, {3: 5}, {3: 1.5}, {3: True}, {"3": 0}, [0]]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                dataset.volume(indices=indices)
        for data in [np.array(["text"]), np.array([object()], dtype=object), np.empty((2, 0, 3)), 1]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                VolumeDataset(data)

    def test_nonfinite_selected_frame_is_rejected_without_hiding_it(self):
        raw = np.ones((2, 3, 4, 2), dtype=np.float32)
        raw[0, 0, 0, 1] = np.nan
        dataset = VolumeDataset(raw)
        np.testing.assert_array_equal(dataset.volume(indices={3: 0}), 1)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            dataset.volume(indices={3: 1})
        for value in [np.inf, np.nan, 1e100]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                VolumeDataset(np.full((2, 2, 2), value)).volume()

    def test_signed_integer_magnitude_does_not_wrap(self):
        raw = np.full((2, 2, 2), np.iinfo(np.int64).min, dtype=np.int64)
        actual = VolumeDataset(raw).volume()
        self.assertTrue(np.all(actual > 0))
        np.testing.assert_allclose(actual, float(2 ** 63))


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "scan.v1"

    def write_pair(self, header, payload):
        Path(str(self.base) + ".hdr").write_text(header, encoding="utf-8")
        Path(str(self.base) + ".cfl").write_bytes(payload)

    def test_cfl_roundtrip_fortran_complex_and_singletons(self):
        shape = (3, 4, 1, 2, 1)
        values = np.arange(math.prod(shape)).reshape(shape)
        raw = (values + 1j * (100 - values)).astype("<c8")
        self.write_pair("# Dimensions\n\n3 4 1 2 1\n", raw.tobytes(order="F"))
        for suffix in ["", ".hdr", ".cfl"]:
            with self.subTest(suffix=suffix):
                loaded = read_cfl(str(self.base) + suffix)
                self.assertIsInstance(loaded, np.memmap)
                self.assertFalse(loaded.flags.writeable)
                np.testing.assert_array_equal(loaded, raw)
        dataset = load_dataset(self.base, spacing=(1, 2, 3))
        self.assertEqual(dataset.name, "scan.v1")
        self.assertEqual(dataset.spatial_shape, (3, 4, 1))
        np.testing.assert_array_equal(dataset.volume("imag", {3: 1}), np.imag(raw[:, :, :, 1, 0]))

    def test_malformed_cfl_headers(self):
        for header in ["", "# dimensions only\n", "2 0 3\n", "2 -1 3\n", "2 2.5 3\n", "2 3 4\n5 6 7\n"]:
            with self.subTest(header=header), self.assertRaises(ValueError):
                self.write_pair(header, b"\0" * 32)
                read_cfl(self.base)

    def test_missing_pair_and_short_or_long_payload(self):
        with self.assertRaises(FileNotFoundError):
            read_cfl(self.base)
        for size in [0, 31, 33, 40]:
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "size mismatch"):
                self.write_pair("2 2 1\n", b"\0" * size)
                read_cfl(self.base)

    def test_npy_load_and_object_pickle_rejection(self):
        path = Path(self.temp.name) / "scan.npy"
        raw = np.arange(24).reshape(2, 3, 4)
        np.save(path, raw)
        dataset = load_dataset(path)
        np.testing.assert_array_equal(dataset.volume(), raw)
        np.save(path, np.array([{"unsafe": "object data"}], dtype=object))
        with self.assertRaises(ValueError):
            load_dataset(path)


class DisplayDataTests(unittest.TestCase):
    def test_robust_contrast_reduces_outlier_influence(self):
        values = np.linspace(0, 100, 10000)
        values[-1] = 1e9
        low, high = robust_clim(values)
        self.assertGreater(low, 0)
        self.assertLess(high, 101)
        self.assertGreater(high, low)

    def test_constant_and_sparse_contrast_ranges(self):
        for value in [0.0, 2.0, -4.0]:
            low, high = robust_clim(np.full((3, 3, 3), value), "real")
            self.assertLess(low, value)
            self.assertGreater(high, value)
        self.assertEqual(robust_clim(np.zeros((2, 2, 2))), (0.0, 0.5))
        sparse = np.zeros(1000)
        sparse[0] = 3
        self.assertEqual(robust_clim(sparse), (0.0, 3.0))
        self.assertEqual(robust_clim(sparse, "angle"), (-math.pi, math.pi))
        self.assertEqual(robust_clim(sparse, "phase"), (-math.pi, math.pi))
        low, high = robust_clim(np.full((2, 2, 2), 1e-12))
        self.assertGreater(low, 0)
        self.assertLess(low, 1e-12)
        self.assertGreater(high, 1e-12)
        self.assertLess(high, 2e-12)

    def test_contrast_rejects_invalid_values(self):
        for data in [[], [np.nan], [1, np.inf], [1 + 2j], ["abc"]]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                robust_clim(data)

    def test_demo_has_finite_asymmetric_spatial_signal(self):
        dataset = make_demo_dataset()
        self.assertIn("Synthetic", dataset.name)
        volume = dataset.volume()
        self.assertEqual(volume.shape, (96, 96, 96))
        self.assertTrue(np.isfinite(volume).all())
        self.assertGreater(volume.max(), 0.5)
        self.assertFalse(np.allclose(volume, volume[::-1]))
        self.assertTrue(np.isfinite(dataset.volume("phase")).all())


if __name__ == "__main__":
    unittest.main()
