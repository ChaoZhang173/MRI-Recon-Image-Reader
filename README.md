# MRI Reconstruction Image Reader

An advanced viewer for Complex Float Library (CFL) data with multi-dimensional navigation capabilities. Designed for MRI reconstruction data visualization and analysis.

## Features

- **Multi-directional slicing**: View data along X, Y, or Z axes
- **90-degree rotation**: Rotate images in 90-degree increments
- **Multi-dimensional navigation**: Navigate through higher-dimensional data (4D, 5D, etc.)
- **Automatic dimension processing**: Intelligent handling of multi-dimensional data
- **Multiple display modes**: Magnitude, phase, real, and imaginary components
- **Interactive controls**: Sliders, buttons, and keyboard shortcuts
- **Auto window/level adjustment**: Automatic contrast optimization
- **Perfect layout**: No overlapping UI elements

## File Structure

```
python_tools/
├── cfl_viewer.py          # Main Python script
├── CFL_Viewer.ipynb       # Jupyter notebook version
├── cfl_reader.py          # CFL file reading utility
└── README.md              # This file
```

## Requirements

### Environment Setup

The viewer requires a specific conda environment. Activate it using:

```bash
source /lustre/nvwulf/projects/KeeGroup-nvwulf/ykee/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/nvwulf/projects/KeeGroup-nvwulf/ykee/mrienv/cones-dev
```

### Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- `cfl_reader.py` (included in this package)

## Usage

### Python Script Version

```bash
# Command line usage
python cfl_viewer.py <path_to_cfl_file>

# Example
python cfl_viewer.py /path/to/data/mri_data
```

Alternatively, you can set the `DEFAULT_FILE_PATH` variable in the script to hardcode a file path.

### Jupyter Notebook Version

1. Open `CFL_Viewer.ipynb` in Jupyter
2. Set the `CFL_FILE_PATH` variable in the first code cell
3. Run all cells
4. Uncomment and run the launcher in the last cell

## Controls

### Mouse Controls
- **Left click on image**: Display pixel value
- **Scroll wheel**: Navigate through slices

### Keyboard Shortcuts
- **Left/Right arrows**: Previous/next slice
- **A**: Auto window/level adjustment
- **R**: Rotate clockwise 90°
- **X/Y/Z**: Switch to X/Y/Z slice direction

### GUI Controls
- **Sliders**: Navigate slices and dimensions
- **Radio buttons**: Change slice direction, display mode, colormap
- **Buttons**: Rotation, auto window/level, reset, navigation

## Data Format

The viewer supports CFL (Complex Float Library) format data files:
- `.cfl` file: Binary data
- `.hdr` file: Header with dimension information

## Examples

### Loading 3D MRI Data
```python
# In Python script
python cfl_viewer.py /data/mri_3d_scan

# In Jupyter notebook
CFL_FILE_PATH = "/data/mri_3d_scan"
viewer = launch_viewer()
```

### Multi-dimensional Data
The viewer automatically processes multi-dimensional data (4D, 5D, etc.) by:
- Keeping the first 3 spatial dimensions (X, Y, Z)
- Providing sliders for up to 2 additional dimensions
- Removing singleton dimensions intelligently

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure the conda environment is activated
2. **File not found**: Check that both `.cfl` and `.hdr` files exist
3. **Memory issues**: Large datasets may require sufficient RAM

### Performance Tips

- Use auto window/level for optimal contrast
- For large datasets, consider the data size before loading
- Use keyboard shortcuts for faster navigation

## Technical Details

### Dimension Processing
- Input data with >3 dimensions is automatically processed
- Up to 2 non-unit dimensions beyond X,Y,Z are preserved
- Singleton dimensions are automatically removed

### Display Modes
- **Magnitude**: `abs(data)`
- **Phase**: `angle(data)` (uses HSV colormap)
- **Real**: `real(data)`
- **Imaginary**: `imag(data)`

### Rotation
- Rotations are applied in 90-degree increments
- Uses NumPy's `rot90` function for efficient rotation

## License

MIT License - See source code for full license text.

## Authors

MRI Research Group

## Version History

- v1.0: Initial release with advanced multi-dimensional navigation
- Perfect layout with all requested features implemented
