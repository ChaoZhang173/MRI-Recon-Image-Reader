#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cflViewer_compare.py — CFL compare viewer with 3 panels: Data1 / Data2 / Diff(A-B)

Layout:
  - 3 images side-by-side: Data1 | Data2 | Diff
  - 3 horizontal colorbars under each image
  - Right control panel + bottom sliders keep the same style as the original viewer

Mouse:
  Right-drag : adjust Window/Level (W/L) for BOTH (Data1&2) and Diff simultaneously
  Left-drag  : horizontal -> change 4th dim (if exists), vertical -> change slice
  Wheel      : prev/next slice (in image axes)

Keys:
  1/2/3      : switch slice axis (x/y/z)
  ↑/↓        : prev/next slice
  ←/→        : prev/next 4th dim (D4, e.g. echoes) if exists
  Ctrl+←/→   : prev/next 5th dim (D5, e.g. motions) if exists
  z / c      : rotate 90° CCW / CW
  a          : toggle Auto W/L (for BOTH main and diff)
  ESC/q      : close
"""

import os, argparse, time
import numpy as np
from collections import OrderedDict

# ---- set a stable backend before importing pyplot ----
import matplotlib
if "MPLBACKEND" not in os.environ:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons

# ---------- perf limits ----------
MOTION_FPS = 60.0              # max mouse motion handling rate
DRAW_FPS   = 30.0              # max main window redraw fps

# ---------- CFL reader ----------
def read_cfl(basepath):
    hdr = basepath + ".hdr"
    cfl = basepath + ".cfl"
    if not (os.path.exists(hdr) and os.path.exists(cfl)):
        raise FileNotFoundError(f"Missing {hdr} or {cfl}")
    with open(hdr, "r") as f:
        line = f.readline()
        while line and line.strip().startswith("#"):
            line = f.readline()
        if not line:
            raise ValueError("HDR has no dims line.")
        dims = [int(x) for x in line.strip().split()]

    n = int(np.prod(dims))
    with open(cfl, "rb") as f:
        raw = np.fromfile(f, dtype=np.float32, count=2*n)
    if raw.size != 2*n:
        raise ValueError(f"Expected {2*n} float32, got {raw.size}")

    arr = raw.reshape((2, *dims), order="F")
    data = arr[0] + 1j*arr[1]
    return data

def squeeze_keep_extras(vol, max_extra=3):
    """
    Squeeze all dimensions of size 1, ensure ndim >= 3.
    Returns: data(ndim>=3), extra_sizes(list), merge_factor(int)
    """
    v = np.squeeze(vol)
    if v.ndim < 3:
        while v.ndim < 3:
            v = v[..., np.newaxis]
    if v.ndim == 3:
        return np.ascontiguousarray(v), [], 1

    base = v.shape[:3]
    tail = list(v.shape[3:])
    if len(tail) <= max_extra:
        return np.ascontiguousarray(v), tail, 1

    keep = tail[:max_extra-1]
    last = tail[max_extra-1:]
    merge = int(np.prod(last))
    extras = keep + [merge]
    newshape = (*base, *keep, merge)
    v2 = np.reshape(v, newshape, order="F")
    return np.ascontiguousarray(v2), extras, merge

def component_view(arr, which):
    if which == "abs":   return np.abs(arr)
    if which == "real":  return np.real(arr)
    if which == "imag":  return np.imag(arr)
    if which == "angle": return np.angle(arr)
    raise ValueError(which)

def rotate_cw_contig(img, k):
    """Rotate 2D array clockwise by k*90 deg, return C-contiguous."""
    k %= 4
    if k == 0:
        out = img
    elif k == 1:   # CW 90
        out = np.flipud(img).T
    elif k == 2:   # 180
        out = np.flipud(np.fliplr(img))
    else:          # CW 270
        out = np.fliplr(img).T
    return np.ascontiguousarray(out)

# ---------- Viewer ----------
class CompareViewer:
    def __init__(self, data1, data2, vox=(1,1,1), title="", cmap="gray",
                 name1="Data1", name2="Data2", diff_name="Diff (A-B)"):
        """
        data1/data2: complex ndarray; dims should match after squeeze_keep_extras()
        """
        self.data1, self.extra_sizes, self.merge_factor = squeeze_keep_extras(data1, max_extra=3)
        self.data2, extra2, merge2 = squeeze_keep_extras(data2, max_extra=3)

        if self.data1.shape != self.data2.shape:
            raise ValueError(f"Shape mismatch after squeeze: data1 {self.data1.shape} vs data2 {self.data2.shape}")
        if extra2 != self.extra_sizes or merge2 != self.merge_factor:
            raise ValueError("Extra dimension layout mismatch after squeeze/merge.")

        self.X, self.Y, self.Z = self.data1.shape[:3]
        self.Nextra = max(0, self.data1.ndim - 3)  # 0..3
        self.extra_idx = [0]*self.Nextra

        self.vox = tuple(float(v) for v in (vox if len(vox)>=3 else (1,1,1)))
        self.cmap_name = cmap
        self.title = title if title else "cflViewer_compare"
        self.name1 = name1
        self.name2 = name2
        self.nameD = diff_name

        # 状态（全局，作用于3张图）
        self.axis = 'z'
        self.slice_idx = self.Z//2
        self.part = "abs"
        self.rot_deg = 0
        self.auto_wl = True

        # W/L: main(1&2共用) + diff(单独)
        self.window_main = 1.0
        self.level_main  = 0.5
        self.window_diff = 1.0
        self.level_diff  = 0.0

        # Angle colormap option
        self.angle_color = False
        self.angle_cmap_name = "hsv"

        # Throttling
        self._last_motion = 0.0
        self._last_draw   = 0.0
        self._draw_interval = 1.0/DRAW_FPS

        # Dragging state
        self._wl_drag = None
        self._wl_dragging = False
        self._scroll_drag = None

        # Diff cache (scheme 2)
        self._diff_cache = OrderedDict()
        self._diff_cache_cap = 24

        # ---- Layout ----
        self._build_gui()
        self._connect_events()
        self._update_all(force=True)

    # ----- GUI -----
    def _build_gui(self):
        self.fig = plt.figure(num=self.title, figsize=(14, 8))

        # --- left 3 panels ---
        left = 0.06
        right_panel = 0.72
        W = right_panel - left - 0.04
        gap = 0.02
        w_each = (W - 2*gap) / 3.0

        img_bottom = 0.28
        img_h = 0.66
        cb_bottom = 0.25
        cb_h = 0.02

        x1 = left
        x2 = left + w_each + gap
        x3 = left + 2*(w_each + gap)

        self.ax1 = self.fig.add_axes([x1, img_bottom, w_each, img_h])
        self.ax2 = self.fig.add_axes([x2, img_bottom, w_each, img_h])
        self.axd = self.fig.add_axes([x3, img_bottom, w_each, img_h])

        self.ax1.set_aspect(self._aspect_for_axis(self.axis))
        self.ax2.set_aspect(self._aspect_for_axis(self.axis))
        self.axd.set_aspect(self._aspect_for_axis(self.axis))

        # colorbars under each image
        self.cax1 = self.fig.add_axes([x1, cb_bottom, w_each, cb_h])
        self.cax2 = self.fig.add_axes([x2, cb_bottom, w_each, cb_h])
        self.caxd = self.fig.add_axes([x3, cb_bottom, w_each, cb_h])

        # --- sliders (same style, width aligned to 3-panel area) ---
        self.ax_slice = self.fig.add_axes([left, 0.16, W, 0.04])
        self.ax_slice.set_xlim(1, self._max_slice() + 1)
        self.slider_slice = Slider(
            self.ax_slice,
            f"Slice ({self.axis.upper()})",
            1,
            self._max_slice() + 1,
            valinit=self.slice_idx + 1,
            valstep=1,
        )

        # Extra dimension sliders (up to 3)
        self.extra_sliders = []
        y0 = 0.10
        for i in range(self.Nextra):
            ax = self.fig.add_axes([left, y0, W, 0.04])
            lab = f"D{4+i}"
            mx  = self.extra_sizes[i]
            sld = Slider(ax, lab, 1, mx, valinit=1, valstep=1)
            self.extra_sliders.append(sld)
            y0 -= 0.06

        # --- Right control panel (same as original) ---
        right = right_panel

        ax_radio = self.fig.add_axes([right, 0.72, 0.25, 0.18])
        ax_radio.set_title("Component")
        self.radio_comp = RadioButtons(ax_radio, ("abs","real","imag","angle"), active=0)

        ax_axis = self.fig.add_axes([right, 0.56, 0.25, 0.12])
        ax_axis.set_title("Slice Axis")
        self.radio_axis = RadioButtons(ax_axis, ("x","y","z"), active=2)

        ax_rot_cw  = self.fig.add_axes([right, 0.50, 0.12, 0.05])
        ax_rot_ccw = self.fig.add_axes([right+0.13, 0.50, 0.12, 0.05])
        self.btn_rot_cw  = Button(ax_rot_cw, "Rotate CW")
        self.btn_rot_ccw = Button(ax_rot_ccw, "Rotate CCW")

        ax_chk = self.fig.add_axes([right, 0.40, 0.25, 0.09])
        self.chk = CheckButtons(ax_chk, ["AutoWL", "AngleColor"], [True, False])

        ax_info = self.fig.add_axes([right, 0.25, 0.25, 0.14]); ax_info.axis("off")
        self.txt_info = ax_info.text(0, 1, "W/L:\n- / -\n- / -", va="top", fontsize=9)

        ax_prev = self.fig.add_axes([right, 0.18, 0.12, 0.05])
        ax_next = self.fig.add_axes([right+0.13, 0.18, 0.12, 0.05])
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")

        # Initialize images + colorbars
        a1, a2, ad = self._current_images2d()
        self.im1 = self.ax1.imshow(a1, cmap=self._cmap_for_part(), origin="upper", interpolation="nearest")
        self.im2 = self.ax2.imshow(a2, cmap=self._cmap_for_part(), origin="upper", interpolation="nearest")
        self.imd = self.axd.imshow(ad, cmap=self._cmap_for_part(), origin="upper", interpolation="nearest")

        self.cbar1 = plt.colorbar(self.im1, cax=self.cax1, orientation="horizontal")
        self.cbar2 = plt.colorbar(self.im2, cax=self.cax2, orientation="horizontal")
        self.cbard = plt.colorbar(self.imd, cax=self.caxd, orientation="horizontal")

        self._apply_wl(force=True)

    def _connect_events(self):
        self.slider_slice.on_changed(self._on_slice_slider)
        for i, sld in enumerate(self.extra_sliders):
            sld.on_changed(lambda val, k=i: self._on_extra_slider(k, val))

        self.radio_comp.on_clicked(self._on_radio_comp)
        self.radio_axis.on_clicked(self._on_radio_axis)
        self.btn_rot_cw.on_clicked(lambda evt: self._rotate(90))
        self.btn_rot_ccw.on_clicked(lambda evt: self._rotate(-90))
        self.chk.on_clicked(self._on_check)
        self.btn_prev.on_clicked(lambda evt: self._step_slice(-1))
        self.btn_next.on_clicked(lambda evt: self._step_slice(+1))

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)

    # ----- helpers -----
    def _img_axes(self):
        return (self.ax1, self.ax2, self.axd)

    def _in_img_axes(self, ax):
        return ax in self._img_axes()

    def _cmap_for_part(self):
        if self.part == "angle":
            return self.angle_cmap_name if self.angle_color else self.cmap_name
        return self.cmap_name

    def _aspect_for_axis(self, axis):
        dx, dy, dz = self.vox
        if axis == 'z':   return dy/dx
        if axis == 'y':   return dz/dx
        if axis == 'x':   return dz/dy
        return 1.0

    def _max_slice(self):
        if self.axis == 'x': return self.X-1
        if self.axis == 'y': return self.Y-1
        return self.Z-1

    def _current_indices(self):
        idx = [slice(None), slice(None), slice(None)]
        if self.axis == 'x':
            idx[0] = self.slice_idx
        elif self.axis == 'y':
            idx[1] = self.slice_idx
        else:
            idx[2] = self.slice_idx
        for e in self.extra_idx:
            idx.append(e)
        return tuple(idx)

    def _diff_cache_key(self):
        return (self.axis, int(self.slice_idx), *[int(e) for e in self.extra_idx])

    def _get_diff_slab(self, idx):
        key = self._diff_cache_key()
        hit = self._diff_cache.get(key, None)
        if hit is not None:
            self._diff_cache.move_to_end(key)
            return hit

        slab1 = self.data1[idx]
        slab2 = self.data2[idx]
        d = slab1 - slab2

        self._diff_cache[key] = d
        self._diff_cache.move_to_end(key)
        while len(self._diff_cache) > self._diff_cache_cap:
            self._diff_cache.popitem(last=False)
        return d

    def _slab_to_img2d(self, slab_complex):
        img = component_view(slab_complex, self.part)
        img = img.T
        k_cw = int((int(self.rot_deg) // 90) % 4)
        img = rotate_cw_contig(img, k_cw)
        return img

    def _current_images2d(self):
        idx = self._current_indices()
        slab1 = self.data1[idx]
        slab2 = self.data2[idx]
        slabd = self._get_diff_slab(idx)

        a1 = self._slab_to_img2d(slab1)
        a2 = self._slab_to_img2d(slab2)
        ad = self._slab_to_img2d(slabd)
        return a1, a2, ad

    def _auto_wl_from(self, a2d):
        mn = float(np.nanmin(a2d))
        mx = float(np.nanmax(a2d))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            mn, mx = 0.0, 1.0
        lev = 0.5*(mx+mn)
        win = (mx-mn)
        return win, lev

    def _auto_wl_main_from_two(self, a1, a2):
        mn1 = float(np.nanmin(a1)); mx1 = float(np.nanmax(a1))
        mn2 = float(np.nanmin(a2)); mx2 = float(np.nanmax(a2))
        mn = np.nanmin([mn1, mn2])
        mx = np.nanmax([mx1, mx2])
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            mn, mx = 0.0, 1.0
        lev = 0.5*(mx+mn)
        win = (mx-mn)
        return win, lev

    def _apply_imshow_common(self, ax, im, a, vmin, vmax):
        im.set_data(a)
        im.set_cmap(self._cmap_for_part())
        im.set_clim(vmin, vmax)
        h, w = a.shape
        im.set_extent((-0.5, w-0.5, h-0.5, -0.5))
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)

    def _apply_titles_and_aspect(self):
        asp = self._aspect_for_axis(self.axis)
        if (self.rot_deg % 180) != 0:
            if asp != 0:
                asp = 1.0 / asp
        for ax in self._img_axes():
            ax.set_aspect(asp)

        k_cw = int((int(self.rot_deg) // 90) % 4)
        status = f"{self.part} — {self.axis.upper()} slice {self.slice_idx+1} — rot {self.rot_deg}° (k={k_cw})"
        self.fig.suptitle(status, fontsize=12, y=0.98)

        self.ax1.set_title(self.name1, fontsize=11)
        self.ax2.set_title(self.name2, fontsize=11)
        self.axd.set_title(self.nameD, fontsize=11)

    def _apply_wl(self, force=False):
        a1, a2, ad = self._current_images2d()

        if self.auto_wl:
            self.window_main, self.level_main = self._auto_wl_main_from_two(a1, a2)
            self.window_diff, self.level_diff = self._auto_wl_from(ad)

        vmin_m = self.level_main - self.window_main/2.0
        vmax_m = self.level_main + self.window_main/2.0
        vmin_d = self.level_diff - self.window_diff/2.0
        vmax_d = self.level_diff + self.window_diff/2.0

        # fallback if invalid
        if (not np.isfinite(vmin_m)) or (not np.isfinite(vmax_m)) or (vmin_m == vmax_m):
            self.window_main, self.level_main = self._auto_wl_main_from_two(a1, a2)
            vmin_m = self.level_main - self.window_main/2.0
            vmax_m = self.level_main + self.window_main/2.0
        if (not np.isfinite(vmin_d)) or (not np.isfinite(vmax_d)) or (vmin_d == vmax_d):
            self.window_diff, self.level_diff = self._auto_wl_from(ad)
            vmin_d = self.level_diff - self.window_diff/2.0
            vmax_d = self.level_diff + self.window_diff/2.0

        self._apply_imshow_common(self.ax1, self.im1, a1, vmin_m, vmax_m)
        self._apply_imshow_common(self.ax2, self.im2, a2, vmin_m, vmax_m)
        self._apply_imshow_common(self.axd, self.imd, ad, vmin_d, vmax_d)

        self._apply_titles_and_aspect()

        try:
            self.cbar1.update_normal(self.im1)
            self.cbar2.update_normal(self.im2)
            self.cbard.update_normal(self.imd)
        except Exception:
            pass

        self.txt_info.set_text(
            "W/L:\n"
            f"main: {self.window_main:.6g} / {self.level_main:.6g}\n"
            f"diff: {self.window_diff:.6g} / {self.level_diff:.6g}"
        )
        self._throttled_draw(force=force)

    def _throttled_draw(self, force=False):
        now = time.monotonic()
        if force or (now - self._last_draw >= self._draw_interval):
            self.fig.canvas.draw_idle()
            self._last_draw = now

    def _update_all(self, force=False):
        self._apply_wl(force=force)

    # ----- callbacks -----
    def _on_slice_slider(self, val):
        idx = int(round(val)) - 1
        idx = max(0, min(self._max_slice(), idx))
        if idx != self.slice_idx:
            self.slice_idx = idx
            self._update_all()

    def _on_extra_slider(self, k, val):
        idx = int(round(val)) - 1
        idx = max(0, min(self.extra_sizes[k]-1, idx))
        if idx != self.extra_idx[k]:
            self.extra_idx[k] = idx
            self._update_all()

    def _on_radio_comp(self, label):
        self.part = str(label)
        self._update_all()

    def _on_radio_axis(self, label):
        self.axis = str(label)
        self.slider_slice.valmin = 1
        self.slider_slice.valmax = self._max_slice()+1
        self.slider_slice.label.set_text(f"Slice ({self.axis.upper()})")
        self.slider_slice.ax.set_xlim(self.slider_slice.valmin, self.slider_slice.valmax)

        mid = (self._max_slice())//2
        self._set_slider_safely(self.slider_slice, mid+1)
        self.slice_idx = mid
        self._update_all(force=True)

    def _rotate(self, delta_deg):
        self.rot_deg = (int(self.rot_deg) + int(delta_deg)) % 360
        self.rot_deg = (self.rot_deg // 90) * 90
        self._update_all()

    def _on_check(self, label):
        if label == "AutoWL":
            self.auto_wl = not self.auto_wl
            self._update_all()
        elif label == "AngleColor":
            self.angle_color = not self.angle_color
            self._update_all()

    def _on_scroll(self, ev):
        if not self._in_img_axes(ev.inaxes):
            return
        step = +1 if getattr(ev, "button", None) == "up" else -1
        self._step_slice(step)

    def _step_slice(self, step):
        new_idx = int(np.clip(self.slice_idx + step, 0, self._max_slice()))
        if new_idx != self.slice_idx:
            self.slice_idx = new_idx
            self._set_slider_safely(self.slider_slice, self.slice_idx+1)
            self._update_all()

    def _step_extra(self, k, step):
        if self.Nextra <= k:
            return
        new_idx = int(np.clip(self.extra_idx[k] + step, 0, self.extra_sizes[k]-1))
        if new_idx != self.extra_idx[k]:
            self.extra_idx[k] = new_idx
            self._set_slider_safely(self.extra_sliders[k], self.extra_idx[k] + 1)
            self._update_all()

    def _on_key(self, ev):
        if ev.key in ("ctrl+left", "control+left"):
            if self.Nextra >= 2:
                self._step_extra(1, -1)
            return
        elif ev.key in ("ctrl+right", "control+right"):
            if self.Nextra >= 2:
                self._step_extra(1, +1)
            return

        if ev.key == "up":
            self._step_slice(-1)
        elif ev.key == "down":
            self._step_slice(+1)
        elif ev.key == "left":
            if self.Nextra >= 1:
                self._step_extra(0, -1)
            else:
                self._step_slice(-1)
        elif ev.key == "right":
            if self.Nextra >= 1:
                self._step_extra(0, +1)
            else:
                self._step_slice(+1)
        elif ev.key == "1":
            self.radio_axis.set_active(0)
        elif ev.key == "2":
            self.radio_axis.set_active(1)
        elif ev.key == "3":
            self.radio_axis.set_active(2)
        elif ev.key == "z":
            self._rotate(-90)
        elif ev.key == "c":
            self._rotate(+90)
        elif ev.key == "a":
            self.auto_wl = not self.auto_wl
            st = self.chk.get_status()[0]
            if st != self.auto_wl:
                self.chk.set_active(0)
            self._update_all()
        elif ev.key in ("escape", "q"):
            plt.close(self.fig)

    def _on_press(self, ev):
        if not self._in_img_axes(ev.inaxes):
            return
        if ev.button == 3:
            self._wl_drag = (ev.x, ev.y,
                             self.window_main, self.level_main,
                             self.window_diff, self.level_diff)
            self._wl_dragging = True
        elif ev.button == 1:
            d40 = self.extra_idx[0] if self.Nextra >= 1 else 0
            self._scroll_drag = (ev.x, ev.y, self.slice_idx, d40)

    def _on_release(self, ev):
        if self._wl_dragging and ev.button == 3:
            self._wl_drag = None
            self._wl_dragging = False
        if self._scroll_drag is not None and ev.button == 1:
            self._scroll_drag = None

    def _on_move(self, ev):
        now = time.monotonic()
        if now - self._last_motion < 1.0/MOTION_FPS:
            return
        self._last_motion = now

        if self._wl_drag is not None:
            x0, y0, win0m, lev0m, win0d, lev0d = self._wl_drag
            dx = (ev.x - x0) if ev.x is not None else 0.0
            dy = (ev.y - y0) if ev.y is not None else 0.0

            rng_m = max(win0m, 1e-6)
            rng_d = max(win0d, 1e-6)

            self.window_main = np.clip(win0m + dx * (rng_m/500.0), 1e-9, 1e12)
            self.level_main  = lev0m + dy * (rng_m/500.0)

            self.window_diff = np.clip(win0d + dx * (rng_d/500.0), 1e-9, 1e12)
            self.level_diff  = lev0d + dy * (rng_d/500.0)

            self._apply_wl()
            return

        if self._scroll_drag is not None and self._in_img_axes(ev.inaxes) and ev.x is not None and ev.y is not None:
            x0, y0, s0, d40 = self._scroll_drag
            dx = ev.x - x0
            dy = ev.y - y0
            step_t = int(np.round(dx / 40.0))
            step_s = int(np.round(-dy / 40.0))

            new_slice = int(np.clip(s0 + step_s, 0, self._max_slice()))
            if new_slice != self.slice_idx:
                self.slice_idx = new_slice
                self._set_slider_safely(self.slider_slice, self.slice_idx+1)

            if self.Nextra >= 1:
                max_t = self.extra_sizes[0] - 1
                new_t = int(np.clip(d40 + step_t, 0, max_t))
                if new_t != self.extra_idx[0]:
                    self.extra_idx[0] = new_t
                    self._set_slider_safely(self.extra_sliders[0], self.extra_idx[0]+1)

            self._update_all()
            return

    def _set_slider_safely(self, slider, val):
        old = slider.eventson
        slider.eventson = False
        try:
            slider.set_val(val)
        finally:
            slider.eventson = old

# ---------- argparse + main ----------
def parse_args():
    ap = argparse.ArgumentParser(description="BART CFL compare viewer: Data1/Data2/Diff")
    ap.add_argument("file1", nargs="?", help="base path WITHOUT extension for dataset 1 (e.g. /path/to/imoutA)")
    ap.add_argument("file2", nargs="?", help="base path WITHOUT extension for dataset 2 (e.g. /path/to/imoutB)")
    ap.add_argument("--file1", dest="file1_flag", help="same as positional file1")
    ap.add_argument("--file2", dest="file2_flag", help="same as positional file2")
    ap.add_argument("--vox", type=float, nargs=3, default=None, help="voxel size dx dy dz")
    ap.add_argument("--title", default="", help="window title")
    ap.add_argument("--cmap", default="gray", help="matplotlib colormap")
    ap.add_argument("--name1", default="Data1", help="label for dataset 1")
    ap.add_argument("--name2", default="Data2", help="label for dataset 2")
    return ap.parse_args()

def main():
    args = parse_args()
    base1 = args.file1_flag or args.file1
    base2 = args.file2_flag or args.file2
    if not base1 or not base2:
        raise SystemExit("Please provide TWO CFL base paths: file1 and file2 (without extension).")

    vol1 = read_cfl(base1)
    vol2 = read_cfl(base2)
    print("raw shape1 from .hdr:", vol1.shape)
    print("raw shape2 from .hdr:", vol2.shape)

    vox = tuple(args.vox) if args.vox is not None else (1.0, 1.0, 1.0)

    t = args.title
    if not t:
        t = f"{os.path.basename(base1)} vs {os.path.basename(base2)} (auto)"

    viewer = CompareViewer(
        vol1, vol2,
        vox=vox,
        title=t,
        cmap=args.cmap,
        name1=args.name1,
        name2=args.name2,
        diff_name="Diff (A-B)"
    )
    plt.show()

if __name__ == "__main__":
    main()
