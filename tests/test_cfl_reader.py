"""CFL interoperability tests with independently packed BART bytes."""

import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from cfl_reader import (get_cfl_info, get_next_line, prepare_viewer_data,
                        read_recon_header, readcfl, writecfl)


class CFLReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'motion-resolved.14236'

    def write_pair(self, header, payload):
        Path(str(self.base) + '.hdr').write_text(header, encoding='utf-8')
        Path(str(self.base) + '.cfl').write_bytes(payload)

    def fixture(self):
        # Non-square XYZ dimensions plus separated echo and motion axes.
        # Packing explicitly, rather than calling the production writer or
        # NumPy reshape, detects a reader/writer pair sharing the same bug.
        shape = (2, 3, 4, 1, 2, 1, 3, 1)
        expected = np.empty(shape, dtype=np.complex64)
        payload = bytearray()
        for motion in range(shape[6]):
            for echo in range(shape[4]):
                for z in range(shape[2]):
                    for y in range(shape[1]):
                        for x in range(shape[0]):
                            real = x + 10 * y + 100 * z + 1000 * echo + 10000 * motion
                            imag = -(real + 7)
                            expected[x, y, z, 0, echo, 0, motion, 0] = complex(real, imag)
                            payload.extend(struct.pack('<ff', real, imag))
        header = '\n# Dimensions\n  # optional comment\n\n'
        header += ' '.join(map(str, shape)) + '\n\n# end\n'
        return header, bytes(payload), expected

    def test_reads_independent_bart_fixture_and_retains_every_axis(self):
        header, payload, expected = self.fixture()
        self.write_pair(header, payload)
        for suffix in ('', '.cfl', '.hdr'):
            for path_type in (str, Path):
                with self.subTest(suffix=suffix, path_type=path_type):
                    path = path_type(str(self.base) + suffix)
                    actual = readcfl(path)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, np.dtype('<c8'))
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(read_recon_header(path), expected.shape)
                    info = get_cfl_info(path)
                    self.assertTrue(info['file_valid'])
                    self.assertEqual(info['total_elements'], expected.size)
                    self.assertEqual(info['file_size_bytes'], len(payload))

    def test_spatial_singleton_does_not_become_an_echo_axis(self):
        self.write_pair(
            '# Dimensions\n2 1 1 3 1\n',
            b''.join(struct.pack('<ff', x + 10 * echo, -x - echo)
                     for echo in range(3) for x in range(2)),
        )
        data = readcfl(self.base)
        self.assertEqual(data.shape, (2, 1, 1, 3, 1))
        self.assertEqual(data[1, 0, 0, 2, 0], 21 - 3j)

    def test_optional_memory_map_is_read_only_and_matches_eager_read(self):
        header, payload, expected = self.fixture()
        self.write_pair(header, payload)
        actual = readcfl(self.base, mmap=True)
        self.assertIsInstance(actual, np.memmap)
        self.assertFalse(actual.flags.writeable)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            actual.flat[0] = 0

    def test_viewer_preparation_preserves_spatial_axes_and_pads_lower_dimensions(self):
        for shape in ((5,), (3, 4), (1, 3, 1, 2, 1, 3), (3, 4, 1, 1)):
            with self.subTest(shape=shape):
                original = np.arange(np.prod(shape)).reshape(shape)
                actual, extras, merge = prepare_viewer_data(original)
                spatial = shape[:3] + (1,) * max(0, 3 - len(shape))
                expected_extras = [size for size in shape[3:] if size > 1]
                self.assertEqual(actual.shape, spatial + tuple(expected_extras))
                self.assertEqual(extras, expected_extras)
                self.assertEqual(merge, 1)
                self.assertEqual(actual.size, original.size)
                self.assertTrue(np.shares_memory(actual, original))

    def test_viewer_preparation_merges_every_frame_in_fortran_order(self):
        shape = (2, 3, 1, 2, 1, 3, 2, 2)
        original = np.arange(np.prod(shape)).reshape(shape)
        actual, extras, merge = prepare_viewer_data(original, max_extra=2)
        self.assertEqual(extras, [2, 12])
        self.assertEqual(merge, 12)
        for a in range(2):
            for b in range(3):
                for c in range(2):
                    for d in range(2):
                        merged_index = b + 3 * c + 6 * d
                        np.testing.assert_array_equal(
                            actual[:, :, :, a, merged_index],
                            original[:, :, :, a, 0, b, c, d],
                        )
        self.assertEqual(actual.size, original.size)

    def test_viewer_preparation_rejects_invalid_inputs(self):
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                prepare_viewer_data(np.ones((2, 3)), max_extra=limit)
        for data in (np.array(1), np.empty((2, 0)), np.array(['text'])):
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                prepare_viewer_data(data)

    def test_writer_produces_canonical_bytes_for_each_path_form(self):
        _, payload, expected = self.fixture()
        for suffix in ('', '.cfl', '.hdr'):
            with self.subTest(suffix=suffix):
                writecfl(Path(str(self.base) + suffix), expected.astype(np.complex128))
                self.assertEqual(Path(str(self.base) + '.cfl').read_bytes(), payload)
                np.testing.assert_array_equal(read_recon_header(self.base), expected.shape)

    def test_writer_handles_noncontiguous_input_in_header_order(self):
        data = (np.arange(24).reshape(4, 6) + 2j)[:, ::2].T
        self.assertFalse(data.flags.c_contiguous)
        expected_bytes = b''.join(
            struct.pack('<ff', data[x, y].real, data[x, y].imag)
            for y in range(data.shape[1]) for x in range(data.shape[0])
        )
        writecfl(self.base, data)
        self.assertEqual(Path(str(self.base) + '.cfl').read_bytes(), expected_bytes)

    def test_only_final_suffix_is_removed_from_nested_extension_names(self):
        _, payload, expected = self.fixture()
        for stem_suffix in ('.cfl', '.hdr'):
            for input_suffix in ('.cfl', '.hdr'):
                with self.subTest(stem_suffix=stem_suffix, input_suffix=input_suffix):
                    base = str(self.base) + stem_suffix
                    path = Path(base + input_suffix)
                    writecfl(path, expected)
                    self.assertEqual(Path(base + '.cfl').read_bytes(), payload)
                    np.testing.assert_array_equal(readcfl(path), expected)
                    np.testing.assert_array_equal(readcfl(path, mmap=True), expected)
                    info = get_cfl_info(path)
                    self.assertTrue(info['file_valid'])
                    self.assertEqual(info['expected_size_bytes'], len(payload))

    def test_rejects_short_and_long_payloads(self):
        for size in (0, 31, 33, 40):
            with self.subTest(size=size):
                self.write_pair('2 2 1\n', b'\0' * size)
                with self.assertRaisesRegex(ValueError, 'payload size mismatch'):
                    readcfl(self.base)
                self.assertFalse(get_cfl_info(self.base)['file_valid'])

    def test_rejects_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            readcfl(self.base)
        Path(str(self.base) + '.hdr').write_text('2 2 1\n', encoding='utf-8')
        with self.assertRaises(FileNotFoundError):
            readcfl(self.base)
        self.assertFalse(get_cfl_info(self.base)['file_valid'])

    def test_rejects_invalid_headers(self):
        for header in ('', '\n# comment\n', '2 0 3\n', '2 -1 3\n', '2 2.5 3\n',
                       '2 nope 3\n', '2 3\n4 5\n', '99999999999999999999\n'):
            with self.subTest(header=header):
                self.write_pair(header, b'\0' * 32)
                with self.assertRaises(ValueError):
                    readcfl(self.base)
        Path(str(self.base) + '.hdr').write_bytes(b'\xff')
        with self.assertRaisesRegex(ValueError, 'UTF-8'):
            readcfl(self.base)

    def test_info_uses_python_integer_product_for_large_headers(self):
        self.write_pair('4294967296 4294967296\n', b'')
        self.assertEqual(get_cfl_info(self.base)['expected_size_bytes'], 8 * 2 ** 64)
        with self.assertRaisesRegex(ValueError, 'payload size mismatch'):
            readcfl(self.base)

    def test_writer_rejects_empty_or_scalar_data_before_creating_files(self):
        for data in (np.empty((2, 0, 3)), np.array(1 + 2j)):
            with self.subTest(shape=data.shape):
                with self.assertRaisesRegex(ValueError, 'positive dimensions'):
                    writecfl(self.base, data)
                self.assertFalse(Path(str(self.base) + '.hdr').exists())
                self.assertFalse(Path(str(self.base) + '.cfl').exists())

    def test_get_next_line_skips_blank_lines_and_comments(self):
        stream = io.StringIO('\n # comment\n\n2 3 4\n\n')
        self.assertEqual(get_next_line(stream), '2 3 4')
        with self.assertRaises(EOFError):
            get_next_line(stream)


if __name__ == '__main__':
    unittest.main()
