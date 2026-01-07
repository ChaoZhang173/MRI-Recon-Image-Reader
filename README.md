# MRI Reconstruction Image Reader

An advanced viewer for Complex Float Library (CFL) data with multi-dimensional navigation capabilities.

## Features

- Multi-directional slicing along X, Y, or Z axes
- 90-degree rotation with both keyboard and button control
- Navigation for higher-dimensional datasets (4D, 5D, 6D)
- Automatic dimension squeezing with extra-dimension sliders
- Multiple display modes: magnitude, phase (angle), real, imaginary
- Auto window/level adjustment for fast contrast tuning
- Keyboard navigation for slices and extra dimensions (D4/D5)
- Colorbar stays visible during manual contrast adjustment
- Optional angle colormap: keep gray by default, switch to color when desired
- Side-by-side comparison viewer (Data1 / Data2 / Diff) for quick volume comparison

## Files

- `cflViewer.py` — official stable viewer with full multi-dimensional GUI controls (single dataset)
- `compareViewer.py` — compare viewer that shows **Data1 / Data2 / Diff (A-B)** side-by-side
- `cfl_viewer.py` — legacy viewer kept for reference only; deprecated due to known issues
- `cfl_reader.py` — helper module for loading CFL data, no longer used

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

# Example: files /.../imout.21025_mresolved_echo-by-echo-gd.{cfl,hdr}
python cflViewer.py /.../imout.21025_mresolved_echo-by-echo-gd
```

The viewer automatically locates the `.cfl` and `.hdr` pair by using the shared base name.

- `--vox dx dy dz` specifies voxel spacing for correct aspect ratios
- `--title` customizes the window title (defaults to the file name)
- `--cmap` chooses a Matplotlib colormap (default: gray)

**Mouse controls**
- Right-drag: adjust window/level
- Left-drag (vertical): change slice
- Left-drag (horizontal, when available): change 4th-dimension index (D4)
- Scroll wheel: move to previous/next slice

**Keyboard controls**
- `1` / `2` / `3`: switch slice axis (X / Y / Z)
- Up / Down arrows: previous / next slice
- Left / Right arrows: previous / next 4th dimension (D4, e.g., echoes); falls back to slice if no extra dim
- Ctrl + Left / Right arrows: previous / next 5th dimension (D5, e.g., motion) when available
- `z` / `c`: rotate 90° counter-clockwise / clockwise
- `a`: toggle auto window/level
- `Esc` / `q`: close the viewer window

**Viewer functions**
- Rotation: use the Rotate CW/CCW buttons or `z`/`c` keys for 90° increments
- Auto W/L: enable the AutoWL checkbox or press `a` to let the viewer track optimal contrast
- Quantitative contrast tuning: the colorbar remains visible even during manual W/L adjustment
- Echo/Motion detection: extra-dimension sliders appear automatically for non-spatial axes (echo, motion, etc.), collapsing singleton dimensions
- Component display: radio buttons switch between magnitude, phase (angle), real, and imaginary views without reloading data
- Angle colormap: angle view uses gray by default; enable the `AngleColor` checkbox (right panel) to switch to a color colormap (e.g., HSV)

### `compareViewer.py` (Data1 / Data2 / Diff)

`compareViewer.py` loads **two** CFL volumes and shows three panels side-by-side:

- **Data1**
- **Data2**
- **Diff (A - B)**

Each panel has its own horizontal colorbar underneath. Data1 and Data2 share the same intensity scale (window/level) to make visual comparison fair, while Diff uses its own scale.

```bash
python compareViewer.py /path/to/cfl_base_A /path/to/cfl_base_B

# With explicit options
python compareViewer.py /path/to/cfl_base_A /path/to/cfl_base_B --vox 0.5 0.5 1.0 --title "A vs B" --cmap gray
```

All navigation and display controls (slice axis, slice index, D4/D5 sliders, rotation, component, AutoWL, AngleColor, etc.) apply to **all three panels simultaneously**.

**Notes**
- Diff is computed on-demand and cached in memory (no disk output) for smooth interaction.
- For datasets with extra dimensions, the indexing is synchronized across all three panels.

### Legacy script

`cfl_viewer.py` is no longer maintained and should not be used for new workflows.
