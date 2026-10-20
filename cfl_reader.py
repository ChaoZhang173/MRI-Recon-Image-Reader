"""
Python implementation of MATLAB readcfl function for reading BART CFL files.

This module provides functionality to read complex floating-point data from
.cfl files along with their dimensions from .hdr files, compatible with the
Berkeley Advanced Reconstruction Toolbox (BART).

Author: Python port of original MATLAB code
Copyright 2016. CBClab, Maastricht University.
Original: 2016 Tim Loderhose (t.loderhose@student.maastrichtuniversity.nl)
"""

import numpy as np
from pathlib import Path


def _base_path(filename_base):
    """Accept a base path or either member of a CFL/HDR pair."""
    base = Path(filename_base).expanduser()
    if base.suffix.lower() in ('.cfl', '.hdr'):
        base = base.with_suffix('')
    return base


def _element_count(dims):
    # Python integers avoid overflow when validating large header dimensions.
    count = 1
    for dim in dims:
        count *= int(dim)
    return count


def readcfl(filename_base, mmap=False):
    """
    Read complex data from CFL file.
    
    Parameters:
    -----------
    filename_base : str or pathlib.Path
        Base path, .cfl filename, or .hdr filename
    mmap : bool
        If True, return a read-only memory map instead of loading the payload.
        
    Returns:
    --------
    data : numpy.ndarray
        Complex64 array in header axis order, including singleton dimensions
        
    Example:
    --------
    >>> data = readcfl('/path/to/reconstruction')
    >>> print(data.shape)
    >>> print(data.dtype)  # Should be complex64
    """
    base = _base_path(filename_base)
    dims = read_recon_header(filename_base)
    filename = Path(str(base) + '.cfl')
    if not filename.is_file():
        raise FileNotFoundError(f"CFL file not found: {filename}")

    dtype = np.dtype('<c8')
    count = _element_count(dims)
    expected_bytes = count * dtype.itemsize
    actual_bytes = filename.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"CFL payload size mismatch: header requires {expected_bytes} bytes, "
            f"found {actual_bytes} in {filename}."
        )
    if mmap:
        return np.memmap(filename, dtype=dtype, mode='r', shape=tuple(dims), order='F')
    # BART stores interleaved real/imaginary float32 values with axis 0
    # varying fastest. Reading complex64 directly preserves each pair.
    with open(filename, 'rb') as fid:
        data = np.fromfile(fid, dtype=dtype, count=count)
    if data.size != count:
        raise ValueError(f"CFL payload changed or was truncated while reading {filename}.")
    return data.reshape(tuple(dims), order='F')


def ensure_spatial_dimensions(data):
    """Preserve input axes, padding missing Y/Z axes with trailing singletons.

    CFL's first three axes are X/Y/Z unless a caller explicitly specifies
    another mapping. Singleton spatial axes must never be squeezed away.
    """
    array = np.asanyarray(data)
    if array.dtype.kind not in 'biufc':
        raise ValueError("Volume data must be a numeric array, without objects.")
    if array.ndim == 0 or any(size == 0 for size in array.shape):
        raise ValueError("Volume data must have at least one nonempty axis.")
    if array.ndim < 3:
        array = array.reshape(array.shape + (1,) * (3 - array.ndim))
    return array


def prepare_viewer_data(data, max_extra=3):
    """Return XYZ data, compact extra sizes, and the last-slider merge factor.

    Only singleton axes after XYZ are removed. When there are more extra axes
    than sliders, trailing extras are combined in Fortran order: the earliest
    merged axis changes fastest. No frames are discarded. Preserve views when
    possible so loading a large CFL does not force a second full-volume copy.
    """
    if (isinstance(max_extra, (bool, np.bool_))
            or not isinstance(max_extra, (int, np.integer)) or max_extra < 1):
        raise ValueError("max_extra must be a positive integer.")
    array = ensure_spatial_dimensions(data)
    singleton_extras = tuple(axis for axis in range(3, array.ndim)
                             if array.shape[axis] == 1)
    if singleton_extras:
        array = np.squeeze(array, axis=singleton_extras)
    extras = list(array.shape[3:])
    if len(extras) <= max_extra:
        return array, extras, 1
    merged = _element_count(extras[max_extra - 1:])
    extras = extras[:max_extra - 1] + [merged]
    return array.reshape(array.shape[:3] + tuple(extras), order='F'), extras, merged


def read_recon_header(filename_base):
    """
    Read reconstruction header to get data dimensions.
    
    Parameters:
    -----------
    filename_base : str or pathlib.Path
        Base path, .cfl filename, or .hdr filename
        
    Returns:
    --------
    dims : numpy.ndarray
        Array of dimensions
    """
    filename = Path(str(_base_path(filename_base)) + '.hdr')
    if not filename.is_file():
        raise FileNotFoundError(f"Header file not found: {filename}")

    try:
        with open(filename, 'r', encoding='utf-8') as fid:
            lines = [line.strip() for line in fid
                     if line.strip() and not line.lstrip().startswith('#')]
    except UnicodeError as exc:
        raise ValueError("CFL header must be UTF-8 text containing dimensions.") from exc
    if len(lines) != 1:
        raise ValueError("CFL header must contain exactly one dimensions line.")
    try:
        dims = [int(token) for token in lines[0].split()]
    except ValueError as exc:
        raise ValueError("CFL dimensions must be positive integers.") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("CFL dimensions must be positive integers.")
    try:
        return np.array(dims, dtype=np.int64)
    except OverflowError as exc:
        raise ValueError("CFL dimensions exceed the supported integer range.") from exc


def get_next_line(fid):
    """
    Get next nonblank, non-comment line from file.
    
    Parameters:
    -----------
    fid : file object
        Open file handle
        
    Returns:
    --------
    line : str
        Next nonblank, non-comment line
    """
    while True:
        line = fid.readline()
        if line == '':
            raise EOFError("Unexpected end of file")
        line = line.strip()
        if line and not line.startswith('#'):
            return line


def writecfl(filename_base, data):
    """
    Write complex data to CFL file format.
    
    Parameters:
    -----------
    filename_base : str or pathlib.Path
        Base path, .cfl filename, or .hdr filename
    data : numpy.ndarray
        Complex array to write
    """
    base = _base_path(filename_base)
    data = np.asarray(data)
    if data.ndim == 0 or any(dim == 0 for dim in data.shape):
        raise ValueError("CFL data must have at least one axis and positive dimensions.")
    data = np.asarray(data, dtype='<c8')

    hdr_filename = str(base) + '.hdr'
    with open(hdr_filename, 'w', encoding='utf-8') as fid:
        fid.write('# Dimensions\n')
        fid.write(' '.join(map(str, data.shape)) + '\n')

    # Complex64 already interleaves real/imaginary values; Fortran order
    # makes the first header dimension vary fastest, as BART requires.
    with open(str(base) + '.cfl', 'wb') as fid:
        fid.write(data.tobytes(order='F'))


def get_cfl_info(filename_base):
    """
    Get information about CFL file without loading the data.
    
    Parameters:
    -----------
    filename_base : str or pathlib.Path
        Base path, .cfl filename, or .hdr filename
        
    Returns:
    --------
    info : dict
        Dictionary containing file information
    """
    base = _base_path(filename_base)
    dims = read_recon_header(filename_base)

    cfl_filename = Path(str(base) + '.cfl')
    file_size = cfl_filename.stat().st_size if cfl_filename.is_file() else 0
    total_elements = _element_count(dims)
    expected_size = total_elements * 8  # interleaved float32 real/imaginary pairs
    
    info = {
        'dimensions': dims,
        'ndims': len(dims),
        'total_elements': total_elements,
        'file_size_bytes': file_size,
        'expected_size_bytes': expected_size,
        'dtype': 'complex64',
        'file_valid': file_size == expected_size
    }
    
    return info


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
        print(f"Reading CFL file: {filename}")
        
        # Get file info
        try:
            info = get_cfl_info(filename)
            print("File Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            if info['file_valid']:
                # Read the data
                data = readcfl(filename)
                print(f"\nData loaded successfully:")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  Min/Max magnitude: {np.abs(data).min():.6f} / {np.abs(data).max():.6f}")
                print(f"  Memory usage: {data.nbytes / (1024**2):.2f} MB")
            else:
                print("Warning: File size mismatch!")
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python cfl_reader.py <filename_base>")
        print("Example: python cfl_reader.py /path/to/reconstruction")
