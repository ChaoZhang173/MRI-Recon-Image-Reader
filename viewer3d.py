#!/usr/bin/env python3
"""Desktop MRI volume viewer. Run without arguments for the synthetic demo."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from pyvistaqt import QtInteractor
except ImportError as exc:
    raise SystemExit(
        "The 3D viewer needs its optional dependencies. Install them with:\n"
        "  python -m pip install -r requirements-3d.txt\n"
        f"Missing dependency: {exc}"
    ) from exc

from volume_data import VolumeDataset, load_dataset, make_demo_dataset, robust_clim
from volume_scene import VolumeScene


class ScientificSpinBox(QtWidgets.QDoubleSpinBox):
    """Keep small reconstruction amplitudes instead of rounding them to zero."""

    def __init__(self):
        super().__init__()
        self.setDecimals(100)
        self.setRange(-1e40, 1e40)
        self.setKeyboardTracking(False)

    def textFromValue(self, value):
        return f"{value:.9g}"

    def valueFromText(self, text):
        return float(text.strip())

    def validate(self, text, position):
        try:
            value = float(text)
            state = QtGui.QValidator.State.Acceptable if math.isfinite(value) and self.minimum() <= value <= self.maximum() else QtGui.QValidator.State.Invalid
        except ValueError:
            state = QtGui.QValidator.State.Intermediate
        return state, text, position

    def stepBy(self, steps):
        self.setValue(self.value() + steps * (abs(self.value()) * 0.05 or 0.05))


class ViewerWindow(QtWidgets.QMainWindow):
    """Controls use original array axis numbers; slice controls use XYZ indices."""

    def __init__(self, dataset: VolumeDataset):
        super().__init__()
        self.setWindowTitle("MRI • 3D Viewer")
        self.resize(1440, 940)
        self._loading = True
        self.dataset = dataset
        self.extra_controls = {}
        self._build_ui()
        self.scene = VolumeScene(self.plotter)
        self.set_dataset(dataset)
        self._loading = False

    def _group(self, title):
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)
        self.controls.addWidget(group)
        return layout

    def _button(self, text, callback, layout):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _slider(self, title, low, high, initial, callback, layout):
        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title)
        value = QtWidgets.QLabel(str(initial))
        value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        row.addWidget(label)
        row.addWidget(value)
        layout.addLayout(row)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(initial)
        slider.setTracking(False)
        slider.setAccessibleName(title)
        slider.valueChanged.connect(lambda v: value.setText(str(v)))
        slider.valueChanged.connect(callback)
        layout.addWidget(slider)
        return slider, value

    def _build_ui(self):
        main = QtWidgets.QSplitter()
        self.setCentralWidget(main)
        sidebar = QtWidgets.QWidget()
        self.controls = QtWidgets.QVBoxLayout(sidebar)
        self.controls.setSpacing(12)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(290)
        main.addWidget(scroll)

        data = self._group("Scan")
        row = QtWidgets.QHBoxLayout()
        self._button("Open scan…", self.open_scan, row)
        self._button("Try demo", lambda: self.set_dataset(make_demo_dataset()), row)
        data.addLayout(row)
        self.scan_info = QtWidgets.QLabel()
        self.scan_info.setWordWrap(True)
        self.scan_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        data.addWidget(self.scan_info)

        geometry = self._group("Geometry")
        form = QtWidgets.QFormLayout()
        self.axes_edit = QtWidgets.QLineEdit("0, 1, 2")
        self.axes_edit.setToolTip("Original array axes to interpret as X, Y, Z (zero-based).")
        form.addRow("XYZ array axes", self.axes_edit)
        geometry.addLayout(form)
        self.known_spacing = QtWidgets.QCheckBox("Voxel spacing is known (mm)")
        geometry.addWidget(self.known_spacing)
        spacing_row = QtWidgets.QHBoxLayout()
        self.spacing_controls = []
        for name in "XYZ":
            spacing_row.addWidget(QtWidgets.QLabel(name))
            control = ScientificSpinBox()
            control.setRange(1e-40, 1e40)
            control.setValue(1)
            control.setKeyboardTracking(False)
            control.setAccessibleName(f"{name} voxel spacing in mm")
            self.spacing_controls.append(control)
            spacing_row.addWidget(control)
        geometry.addLayout(spacing_row)
        self.known_spacing.toggled.connect(
            lambda checked: [control.setEnabled(checked) for control in self.spacing_controls]
        )
        self._button("Apply geometry", self.apply_geometry, geometry)
        note = QtWidgets.QLabel("X/Y/Z are data axes. Anatomical orientation is not inferred from CFL.")
        note.setWordWrap(True)
        geometry.addWidget(note)

        selection = self._group("Volume selection")
        self.extra_layout = QtWidgets.QFormLayout()
        selection.addLayout(self.extra_layout)
        self.component = QtWidgets.QComboBox()
        for label, value in [("Magnitude", "abs"), ("Real", "real"), ("Imaginary", "imag"), ("Phase (radians)", "angle")]:
            self.component.addItem(label, value)
        self.component.currentIndexChanged.connect(self._refresh_volume)
        selection.addWidget(self.component)

        appearance = self._group("Appearance")
        self.auto_contrast = QtWidgets.QCheckBox("Auto contrast on volume change")
        self.auto_contrast.setChecked(True)
        appearance.addWidget(self.auto_contrast)
        contrast = QtWidgets.QFormLayout()
        self.minimum = ScientificSpinBox()
        self.maximum = ScientificSpinBox()
        for control in (self.minimum, self.maximum):
            control.valueChanged.connect(self._appearance_changed)
        contrast.addRow("Display minimum", self.minimum)
        contrast.addRow("Display maximum", self.maximum)
        appearance.addLayout(contrast)
        self._button("Fit contrast", self.fit_contrast, appearance)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["gray", "magma", "viridis", "coolwarm", "twilight"])
        self.cmap.currentTextChanged.connect(self._appearance_changed)
        appearance.addWidget(self.cmap)
        self.blend = QtWidgets.QComboBox()
        self.blend.addItem("Volume rendering", "composite")
        self.blend.addItem("Maximum intensity projection", "maximum")
        self.blend.currentIndexChanged.connect(self._appearance_changed)
        appearance.addWidget(self.blend)
        self.opacity, _ = self._slider("3D opacity (%)", 0, 100, 35, self._appearance_changed, appearance)
        self.threshold, _ = self._slider("Hide lowest intensity range (%)", 0, 99, 10, self._appearance_changed, appearance)

        slices = self._group("Slices · zero-based indices")
        self.slice_sliders = []
        self.slice_labels = []
        for axis, name in enumerate("XYZ"):
            slider, label = self._slider(name, 0, 1, 0, lambda value, a=axis: self._slice_changed(a, value), slices)
            self.slice_sliders.append(slider)
            self.slice_labels.append(label)

        cutaway = self._group("3D cutaway")
        self.cut_enabled = QtWidgets.QCheckBox("Cut through the volume")
        self.cut_enabled.toggled.connect(self._cut_changed)
        cutaway.addWidget(self.cut_enabled)
        self.cut_axis = QtWidgets.QComboBox()
        self.cut_axis.addItems(["X", "Y", "Z"])
        self.cut_axis.currentIndexChanged.connect(self._cut_changed)
        cutaway.addWidget(self.cut_axis)
        self.cut_position, _ = self._slider("Position (%)", 0, 100, 50, self._cut_changed, cutaway)
        self.cut_flip = QtWidgets.QCheckBox("Keep the opposite side")
        self.cut_flip.toggled.connect(self._cut_changed)
        cutaway.addWidget(self.cut_flip)

        actions = self._group("View")
        self._button("Reset cameras", lambda: self.scene.reset_views(), actions)
        self._button("Save view as PNG…", self.save_view, actions)
        instructions = QtWidgets.QLabel("Drag to rotate · Shift-drag to pan\nWheel to zoom · Reset restores slice views")
        instructions.setWordWrap(True)
        actions.addWidget(instructions)
        self.controls.addStretch()

        frame = QtWidgets.QFrame()
        render_layout = QtWidgets.QVBoxLayout(frame)
        render_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(frame, shape=(2, 2), auto_update=False, multi_samples=0)
        render_layout.addWidget(self.plotter.interactor)
        main.addWidget(frame)
        main.setSizes([320, 1120])
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        self.statusBar().showMessage("Ready")
        open_action = QtGui.QAction("Open scan", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_scan)
        self.addAction(open_action)

    def _report_error(self, title, exc):
        self.statusBar().showMessage(str(exc))
        QtWidgets.QMessageBox.warning(self, title, str(exc))

    def set_dataset(self, dataset):
        # Validate before replacing the currently displayed scan.
        initial_volume = dataset.volume(component=self.component.currentData())
        if any(value < 1e-40 or value > 1e40 for value in dataset.spacing):
            raise ValueError("Voxel spacing must be between 1e-40 and 1e40 in this viewer.")
        self._loading = True
        self.dataset = dataset
        self.setWindowTitle(f"MRI • 3D Viewer — {dataset.name}")
        self.axes_edit.setText(", ".join(str(axis) for axis in dataset.spatial_axes))
        self.known_spacing.setChecked(dataset.units == "mm")
        for control, value in zip(self.spacing_controls, dataset.spacing):
            control.setValue(value)
            control.setEnabled(dataset.units == "mm")
        while self.extra_layout.rowCount():
            self.extra_layout.removeRow(0)
        self.extra_controls = {}
        for axis, size in zip(dataset.extra_axes, dataset.extra_sizes):
            control = QtWidgets.QSpinBox()
            control.setRange(0, size - 1)
            control.setKeyboardTracking(False)
            control.valueChanged.connect(self._refresh_volume)
            self.extra_layout.addRow(f"Array axis {axis} (0–{size - 1})", control)
            self.extra_controls[axis] = control
        if not self.extra_controls:
            self.extra_layout.addRow(QtWidgets.QLabel("One volume; no extra dimensions"))
        for slider, label, size in zip(self.slice_sliders, self.slice_labels, dataset.spatial_shape):
            slider.setRange(0, size - 1)
            slider.setValue(size // 2)
            slider.setEnabled(size > 1)
            label.setText(str(size // 2))
        spacing_text = " × ".join(f"{value:g}" for value in dataset.spacing)
        self.scan_info.setText(
            f"{dataset.name}\nXYZ: {' × '.join(map(str, dataset.spatial_shape))}\n"
            + (f"Spacing: {spacing_text} mm" if dataset.units == "mm" else "Spacing unknown · voxel coordinates")
        )
        self.cut_enabled.setChecked(False)
        self._loading = False
        self._display_volume(initial_volume, reset=True)
        self._remember_selection()

    def _display_volume(self, volume, reset=False):
        self._loading = True
        if self.auto_contrast.isChecked() or reset or self.minimum.value() >= self.maximum.value():
            low, high = robust_clim(volume, self.component.currentData())
            self.minimum.setValue(low)
            self.maximum.setValue(high)
        self._loading = False
        self.scene.set_volume(
            volume, tuple(self.dataset.spacing), units=self.dataset.units,
            component=self.component.currentData(),
            clim=(self.minimum.value(), self.maximum.value()), reset_camera=reset,
        )
        self._appearance_changed()
        self._cut_changed()
        for axis, slider in enumerate(self.slice_sliders):
            self.scene.set_slice(axis, slider.value())
        self.statusBar().showMessage(
            f"{self.dataset.name} · {self.component.currentText()} · "
            "Original voxel values; display contrast does not modify the data"
        )

    def _refresh_volume(self, *_):
        if self._loading:
            return
        try:
            indices = {axis: control.value() for axis, control in self.extra_controls.items()}
            volume = self.dataset.volume(self.component.currentData(), indices=indices)
            self._display_volume(volume)
            self._remember_selection()
        except (ValueError, OSError, MemoryError) as exc:
            self._loading = True
            self.component.setCurrentIndex(self.component.findData(self._displayed_component))
            for axis, control in self.extra_controls.items():
                control.setValue(self._displayed_indices.get(axis, 0))
            self._loading = False
            self._report_error("Cannot display this volume", exc)

    def _remember_selection(self):
        self._displayed_component = self.component.currentData()
        self._displayed_indices = {axis: control.value() for axis, control in self.extra_controls.items()}

    def _appearance_changed(self, *_):
        if self._loading:
            return
        low, high = self.minimum.value(), self.maximum.value()
        if low >= high:
            self.statusBar().showMessage("Display minimum must be smaller than display maximum.")
            return
        self.scene.set_appearance(
            clim=(low, high), opacity=self.opacity.value() / 100,
            threshold=self.threshold.value() / 100, cmap=self.cmap.currentText(),
            blend=self.blend.currentData(),
        )

    def fit_contrast(self):
        low, high = robust_clim(self.scene.volume, self.component.currentData())
        self._loading = True
        self.minimum.setValue(low)
        self.maximum.setValue(high)
        self._loading = False
        self._appearance_changed()

    def _slice_changed(self, axis, value):
        if not self._loading:
            self.scene.set_slice(axis, value)

    def _cut_changed(self, *_):
        if not self._loading:
            self.scene.set_cutaway(
                self.cut_enabled.isChecked(), axis=self.cut_axis.currentIndex(),
                fraction=self.cut_position.value() / 100, flip=self.cut_flip.isChecked(),
            )

    def _geometry(self):
        axes = tuple(int(part.strip()) for part in self.axes_edit.text().split(","))
        spacing = tuple(control.value() for control in self.spacing_controls) if self.known_spacing.isChecked() else None
        return axes, spacing

    def apply_geometry(self):
        try:
            axes, spacing = self._geometry()
            self.set_dataset(VolumeDataset(self.dataset.data, spatial_axes=axes, spacing=spacing, name=self.dataset.name))
        except (ValueError, OSError, MemoryError) as exc:
            self._report_error("Check the geometry", exc)

    def open_scan(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open MRI reconstruction", "", "MRI arrays (*.cfl *.hdr *.npy);;All files (*)"
        )
        if not path:
            return
        try:
            # New files must not silently inherit the demo's known millimetre spacing.
            self.set_dataset(load_dataset(path))
        except (ValueError, OSError, MemoryError) as exc:
            self._report_error("Cannot open this scan", exc)

    def save_view(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the four views", "mri-3d-view.png", "PNG image (*.png)")
        if path:
            path = str(Path(path).with_suffix(".png"))
            try:
                self.plotter.screenshot(path)
                self.statusBar().showMessage(f"Saved {path}")
            except (ValueError, OSError) as exc:
                self._report_error("Could not save the view", exc)

    def closeEvent(self, event):
        self.plotter.close()
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        # Native OpenGL surfaces need one render after the window is exposed.
        QtCore.QTimer.singleShot(0, self.plotter.render)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="CFL base path, .cfl/.hdr, or NumPy .npy file")
    parser.add_argument("--file", dest="file_path", help="Alternative to the positional path")
    parser.add_argument("--demo", action="store_true", help="Show a synthetic MRI-like phantom")
    parser.add_argument("--vox", type=float, nargs=3, metavar=("DX", "DY", "DZ"), help="XYZ voxel spacing in mm; otherwise use voxel coordinates")
    parser.add_argument("--spatial-axes", type=int, nargs=3, default=(0, 1, 2), metavar=("X", "Y", "Z"), help="Original zero-based array axes for XYZ (default: 0 1 2)")
    args = parser.parse_args(argv)
    if args.path and args.file_path:
        parser.error("Use either a positional path or --file, not both.")
    if args.demo and (args.path or args.file_path):
        parser.error("Use --demo or a file path, not both.")
    try:
        dataset = load_dataset(args.path or args.file_path, spatial_axes=args.spatial_axes, spacing=args.vox) if args.path or args.file_path else make_demo_dataset()
    except (ValueError, OSError, MemoryError) as exc:
        parser.error(str(exc))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("MRI 3D Viewer")
    try:
        window = ViewerWindow(dataset)
    except (ValueError, OSError, MemoryError) as exc:
        parser.error(str(exc))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
