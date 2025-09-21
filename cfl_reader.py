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
import os


def readcfl(filename_base):
    """
    Read complex data from CFL file.
    
    Parameters:
    -----------
    filename_base : str
        Path and filename of cfl file (without extension)
        
    Returns:
    --------
    data : numpy.ndarray
        Complex array with the reconstructed data
        
    Example:
    --------
    >>> data = readcfl('/path/to/reconstruction')
    >>> print(data.shape)
    >>> print(data.dtype)  # Should be complex64
    """
    # Read dimensions from header file
    dims = read_recon_header(filename_base)
    
    # Read complex data from CFL file
    filename = filename_base + '.cfl'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CFL file not found: {filename}")
    
    # Read binary data as float32
    with open(filename, 'rb') as fid:
        # Read interleaved real and imaginary parts
        data_size = int(np.prod([2] + list(dims)))
        data_r_i = np.frombuffer(fid.read(), dtype=np.float32, count=data_size)
    
    # Reshape to [2, dims...]
    data_r_i = data_r_i.reshape([2] + list(dims))
    
    # Create complex array
    # data_r_i[0, ...] contains real parts, data_r_i[1, ...] contains imaginary parts
    data = data_r_i[0, ...] + 1j * data_r_i[1, ...]
    
    return data.astype(np.complex64)


def read_recon_header(filename_base):
    """
    Read reconstruction header to get data dimensions.
    
    Parameters:
    -----------
    filename_base : str
        Path and filename base (without extension)
        
    Returns:
    --------
    dims : numpy.ndarray
        Array of dimensions
    """
    filename = filename_base + '.hdr'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Header file not found: {filename}")
    
    with open(filename, 'r') as fid:
        line = get_next_line(fid)
        # Parse dimensions from the line
        dims = np.array([int(x) for x in line.split()])
    
    return dims


def get_next_line(fid):
    """
    Get next non-comment line from file.
    
    Parameters:
    -----------
    fid : file object
        Open file handle
        
    Returns:
    --------
    line : str
        Next non-comment line
    """
    while True:
        line = fid.readline().strip()
        if not line:
            raise EOFError("Unexpected end of file")
        if not line.startswith('#'):
            return line


def writecfl(filename_base, data):
    """
    Write complex data to CFL file format.
    
    Parameters:
    -----------
    filename_base : str
        Path and filename base (without extension)
    data : numpy.ndarray
        Complex array to write
    """
    # Write header file
    hdr_filename = filename_base + '.hdr'
    with open(hdr_filename, 'w') as fid:
        fid.write('# Dimensions\n')
        fid.write(' '.join(map(str, data.shape)) + '\n')
    
    # Write CFL file
    cfl_filename = filename_base + '.cfl'
    
    # Convert to complex64 if needed
    if data.dtype != np.complex64:
        data = data.astype(np.complex64)
    
    # Interleave real and imaginary parts
    data_interleaved = np.empty(data.shape + (2,), dtype=np.float32)
    data_interleaved[..., 0] = data.real
    data_interleaved[..., 1] = data.imag
    
    # Reshape to match MATLAB format [2, dims...]
    data_reshaped = np.moveaxis(data_interleaved, -1, 0)
    
    # Write binary data
    with open(cfl_filename, 'wb') as fid:
        data_reshaped.tobytes('C')
        fid.write(data_reshaped.tobytes())


def get_cfl_info(filename_base):
    """
    Get information about CFL file without loading the data.
    
    Parameters:
    -----------
    filename_base : str
        Path and filename base (without extension)
        
    Returns:
    --------
    info : dict
        Dictionary containing file information
    """
    dims = read_recon_header(filename_base)
    
    cfl_filename = filename_base + '.cfl'
    file_size = os.path.getsize(cfl_filename) if os.path.exists(cfl_filename) else 0
    expected_size = int(np.prod(dims)) * 2 * 4  # 2 for complex, 4 bytes per float32
    
    info = {
        'dimensions': dims,
        'ndims': len(dims),
        'total_elements': int(np.prod(dims)),
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