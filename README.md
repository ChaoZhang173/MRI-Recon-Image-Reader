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
- Optional normalization in `compareViewer.py` to scale Data1/Data2 by each displayed image max

## Files

- `cflViewer.py` — official stable viewer with full multi-dimensional GUI controls (single dataset)
- `compareViewer.py` — compare viewer that shows **Data1 / Data2 / Diff (A-B)** side-by-side
- `viewer3d.py` — standalone 3D volume/MIP viewer with three orthogonal slices
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
- Left / Right arrows: previous / next slice
- Up / Down arrows: next / previous 4th dimension (D4, e.g., echoes); falls back to next / previous slice if no extra dim
- Ctrl + Up / Down arrows: next / previous 5th dimension (D5, e.g., motion) when available
- `z` / `c`: rotate 90° counter-clockwise / clockwise
- `w`: toggle auto window/level
- `a`: toggle `AngleColor`
- `Esc`: close the viewer window

**Viewer functions**
- Rotation: use the Rotate CW/CCW buttons or `z`/`c` keys for 90° increments
- Auto W/L: enable the AutoWL checkbox or press `w` to let the viewer track optimal contrast
- Quantitative contrast tuning: the colorbar remains visible even during manual W/L adjustment
- Echo/Motion detection: extra-dimension sliders appear automatically for non-spatial axes (echo, motion, etc.), collapsing singleton dimensions
- Component display: radio buttons switch between magnitude, phase (angle), real, and imaginary views without reloading data
- Angle colormap: angle view uses gray by default; enable the `AngleColor` checkbox (right panel) or press `a` to switch to a color colormap (e.g., HSV)

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

All navigation and display controls (slice axis, slice index, D4/D5 sliders, rotation, component, AutoWL, Normalize, AngleColor, etc.) apply to **all three panels simultaneously**.

**Keyboard controls**
- `1` / `2` / `3`: switch slice axis (X / Y / Z)
- Left / Right arrows: previous / next slice
- Up / Down arrows: next / previous 4th dimension (D4, e.g., echoes); falls back to next / previous slice if no extra dim
- Ctrl + Up / Down arrows: next / previous 5th dimension (D5, e.g., motion) when available
- `z` / `c`: rotate 90° counter-clockwise / clockwise
- `w`: toggle auto window/level for Data1/Data2 and Diff
- `n`: toggle `Normalize`
- `a`: toggle `AngleColor`
- `Esc` / `q`: close the viewer window

**Compare viewer functions**
- Shared navigation: component, slice, extra-dimension index, rotation, and axis changes stay synchronized across Data1, Data2, and Diff
- Shared contrast: Data1 and Data2 share one window/level so comparisons stay visually fair, while Diff keeps an independent scale
- Normalize: enable the `Normalize` checkbox or press `n` to divide the currently displayed Data1 and Data2 images by their own displayed-image maxima before rendering; the view updates immediately
- Angle colormap: enable `AngleColor` or press `a` to use a color colormap when the selected component is `angle`

**Notes**
- Diff is computed on-demand and cached in memory (no disk output) for smooth interaction.
- For datasets with extra dimensions, the indexing is synchronized across all three panels.

### Legacy script

`cfl_viewer.py` is no longer maintained and should not be used for new workflows.

## Standalone 3D viewer

`viewer3d.py` adds volume rendering/MIP, three orthogonal slices, complex
components, extra-dimension selection, contrast, opacity, cutaway, and PNG export.
It runs on an ordinary monitor; the existing 2D viewers remain independent.

Use Python 3.10+ and a desktop with working graphics. From the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-3d.txt
python viewer3d.py --demo
python viewer3d.py /path/to/reconstruction --vox 0.5 0.5 1.0 --spatial-axes 0 1 2
```

Input accepts a CFL/HDR base path, `.cfl`/`.hdr` filename, or numeric `.npy` file.
The demo is a generated phantom, not a patient MRI. On macOS, once `.venv` is
installed, double-click `Launch 3D Viewer.command`, then choose **Open scan**.
On Windows, activate with `.venv\Scripts\activate` instead.

- `--spatial-axes` gives original, zero-based array axes in displayed X/Y/Z order
  (default `0 1 2`). Other non-singleton dimensions get separate selectors.
- `--vox` gives X/Y/Z spacing in mm; omitted spacing uses voxel coordinates.
  The viewer labels data axes and does not infer anatomical orientation.
- Singleton spatial axes remain 2D; the viewer does not invent volume thickness.
- Slice sliders use zero-based indices. Opacity, threshold, and cutaway affect
  the 3D volume; contrast and colormap also apply to the slices.
- Drag to rotate, **Shift-drag** to pan, and use the wheel to zoom in any panel.
  **Reset cameras** restores orthogonal views; **Save view as PNG** saves all four.
- Edit axes/spacing under **Geometry**, then click **Apply geometry**.

Run tests with the optional 3D environment active:

```bash
python -m unittest discover -s tests -v
# Include native GUI/pixel checks on a desktop with working graphics:
MRI_VIEWER_GUI_TESTS=1 python -m unittest discover -s tests -v
```

Default tests check data integrity, exact slices, geometry, and display state
with rendering suppressed. GUI tests additionally render and inspect the views.
