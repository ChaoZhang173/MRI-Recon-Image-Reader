# MRI Reconstruction Image Reader

An advanced viewer for Complex Float Library (CFL) data with multi-dimensional navigation capabilities.

## Features

- Multi-directional slicing along X, Y, or Z axes
- 90-degree rotation with both keyboard and button control
- Navigation for higher-dimensional datasets (4D, 5D, 6D)
- Automatic dimension squeezing with extra-dimension sliders
- Multiple display modes: magnitude, phase, real, imaginary
- Auto window/level adjustment for fast contrast tuning

## Files

- `cflViewer.py` — official stable viewer with full multi-dimensional GUI controls (use this script)
- `cfl_viewer.py` — legacy viewer kept for reference only; deprecated due to known issues
- `cfl_reader.py` — helper module for loading CFL data

## Environment

- Python 3.8 or newer
- NumPy
- Matplotlib

On nvwulf systems activate the prepared environment before running the viewer:

```bash
source /lustre/nvwulf/projects/KeeGroup-nvwulf/ykee/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/nvwulf/projects/KeeGroup-nvwulf/ykee/mrienv/cones-dev
```

## Usage

### `cflViewer.py` (stable release)

`cflViewer.py` accepts either a positional base path or the `--file` flag. Provide the path without the `.cfl`/`.hdr` extensions.

```bash
python cflViewer.py /path/to/cfl_base
# Alternative with explicit options
python cflViewer.py --file /path/to/cfl_base --vox 0.5 0.5 1.0 --title "Subject 01" --cmap plasma
```

- `--vox dx dy dz` specifies voxel spacing for correct aspect ratios
- `--title` customizes the window title (defaults to the file name)
- `--cmap` chooses a Matplotlib colormap (default: gray)

### Legacy script

`cfl_viewer.py` is no longer maintained and should not be used for new workflows.
