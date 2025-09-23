#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cflViewer.py — CFL viewer (stable) with X/Y/Z slicing, rotation, auto W/L, extra-dim sliders.

Mouse:
  Right-drag : adjust Window/Level (W/L)
  Left-drag  : horizontal -> change 4th dim (if exists), vertical -> change slice
  Wheel      : prev/next slice (in image axes)

Keys:
  x/y/z      : switch slice axis
  ←/→        : prev/next slice
  e / q      : rotate 90° CW / CCW
  a          : toggle Auto W/L
  ESC        : close
"""

import os, argparse, time
import numpy as np

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
MOTION_FPS = 60.0              # 鼠标移动处理最大帧率
DRAW_FPS   = 30.0              # 主窗口重绘最大帧率

# ---------- CFL reader ----------
def read_cfl(basepath):
    hdr = basepath + ".hdr"
    cfl = basepath + ".cfl"
    if not (os.path.exists(hdr) and os.path.exists(cfl)):
        raise FileNotFoundError(f"Missing {hdr} or {cfl}")
    # 读维度（第一行非注释）
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
    压缩所有为1的维度，确保 >=3D。
    返回：data(ndim>=3), extra_sizes(list), merge_factor(int, 把第(4+max_extra)以后的维度合并到最后一个slider)
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

# ---------- Viewer ----------
class Viewer:
    def __init__(self, data, vox=(1,1,1), title="", cmap="gray"):
        """
        data: complex ndarray, ndim>=3; shape:(X,Y,Z, D4, D5, ...)
        """
        self.data, self.extra_sizes, self.merge_factor = squeeze_keep_extras(data, max_extra=3)
        self.X, self.Y, self.Z = self.data.shape[:3]
        self.Nextra = max(0, self.data.ndim - 3)  # 0..3
        self.extra_idx = [0]*self.Nextra

        self.vox = tuple(float(v) for v in (vox if len(vox)>=3 else (1,1,1)))
        self.cmap_name = cmap
        self.title = title if title else "cflViewer"

        # 状态
        self.axis = 'z'            # 'x'|'y'|'z'
        self.slice_idx = self.Z//2 # 当前切片编号
        self.part = "abs"          # 'abs'|'real'|'imag'|'angle'
        self.rot_deg = 0           # 0/90/180/270
        self.auto_wl = True        # 自动W/L
        self.window = 1.0
        self.level  = 0.5

        # 节流
        self._last_motion = 0.0
        self._last_draw   = 0.0
        self._draw_interval = 1.0/DRAW_FPS

        # 拖动状态
        self._wl_drag = None        # 右键 W/L 拖动 (x0, y0, win0, lev0)
        self._wl_dragging = False
        self._scroll_drag = None    # 左键 切换slice/第4维 (x0, y0, slice0, d40)

        # ---- 布局 ----
        self._build_gui()
        self._connect_events()
        self._update_all(force=True)

    # ----- GUI -----
    def _build_gui(self):
        self.fig = plt.figure(num=self.title, figsize=(12, 8))
        self.ax_img = self.fig.add_axes([0.06, 0.25, 0.58, 0.70])
        self.ax_img.set_aspect(self._aspect_for_axis(self.axis))

        # 颜色条
        self.ax_cbar = self.fig.add_axes([0.65, 0.25, 0.015, 0.70])

        # 主切片 slider（底部横向）
        self.ax_slice = self.fig.add_axes([0.06, 0.16, 0.58, 0.04])
        self.slider_slice = Slider(self.ax_slice, f"Slice ({self.axis.upper()})",
                                   1, self._max_slice()+1, valinit=self.slice_idx+1, valstep=1)

        # 额外维度 slider（最多3个，逐行放；第5维自然会出现第二个slider）
        self.extra_sliders = []
        y0 = 0.10
        for i in range(self.Nextra):
            ax = self.fig.add_axes([0.06, y0, 0.58, 0.04])
            lab = f"D{4+i}"
            mx  = self.extra_sizes[i]
            sld = Slider(ax, lab, 1, mx, valinit=1, valstep=1)
            self.extra_sliders.append(sld)
            y0 -= 0.06

        # 右侧控制面板
        right = 0.72
        # 1) 选择分量
        ax_radio = self.fig.add_axes([right, 0.72, 0.25, 0.18])
        ax_radio.set_title("Component")
        self.radio_comp = RadioButtons(ax_radio, ("abs","real","imag","angle"), active=0)

        # 2) 切片方向
        ax_axis = self.fig.add_axes([right, 0.56, 0.25, 0.12])
        ax_axis.set_title("Slice Axis")
        self.radio_axis = RadioButtons(ax_axis, ("x","y","z"), active=2)

        # 3) 旋转按钮
        ax_rot_cw  = self.fig.add_axes([right, 0.50, 0.12, 0.05])
        ax_rot_ccw = self.fig.add_axes([right+0.13, 0.50, 0.12, 0.05])
        self.btn_rot_cw  = Button(ax_rot_cw, "Rotate CW")
        self.btn_rot_ccw = Button(ax_rot_ccw, "Rotate CCW")

        # 4) AutoWL 勾选
        ax_chk = self.fig.add_axes([right, 0.42, 0.25, 0.06])
        self.chk = CheckButtons(ax_chk, ["AutoWL"], [True])

        # 5) W/L 信息（只显示）
        ax_info = self.fig.add_axes([right, 0.25, 0.25, 0.14]); ax_info.axis("off")
        self.txt_info = ax_info.text(0, 1, "W/L: - / -", va="top", fontsize=9)

        # 6) 前后切片按钮
        ax_prev = self.fig.add_axes([right, 0.18, 0.12, 0.05])
        ax_next = self.fig.add_axes([right+0.13, 0.18, 0.12, 0.05])
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")

        # 初始化图像和colorbar
        img = self._current_image2d()
        self.im = self.ax_img.imshow(img, cmap=self._cmap_for_part(),
                                     origin="upper", interpolation="nearest")
        self.cbar = plt.colorbar(self.im, cax=self.ax_cbar)
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
    def _cmap_for_part(self):
        return "hsv" if self.part == "angle" else self.cmap_name

    def _aspect_for_axis(self, axis):
        # imshow aspect = (y_scale/x_scale). 选择对应体素尺寸比。
        dx, dy, dz = self.vox
        if axis == 'z':   return dy/dx
        if axis == 'x':   return dy/dz  # 显示 (Y,Z)
        if axis == 'y':   return dx/dz  # 显示 (X,Z)
        return 1.0

    def _max_slice(self):
        if self.axis == 'x': return self.X-1
        if self.axis == 'y': return self.Y-1
        return self.Z-1

    def _current_indices(self):
        # 构造切片索引（含额外维度）
        idx = [slice(None), slice(None), slice(None)]
        if self.axis == 'x':
            idx[0] = self.slice_idx
        elif self.axis == 'y':
            idx[1] = self.slice_idx
        else:
            idx[2] = self.slice_idx
        for k, e in enumerate(self.extra_idx):
            idx.append(e)
        return tuple(idx)

    def _current_image2d(self):
        idx = self._current_indices()
        slab = self.data[idx]
        img = component_view(slab, self.part)
        k = (self.rot_deg % 360)//90
        if k: img = np.rot90(img, k=k)
        return img

    def _auto_wl_from(self, a2d):
        mn = float(np.nanmin(a2d))
        mx = float(np.nanmax(a2d))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            mn, mx = 0.0, 1.0
        self.level = 0.5*(mx+mn)
        self.window = (mx-mn)

    def _apply_wl(self, force=False):
        a = self._current_image2d()
        if self.auto_wl:
            self._auto_wl_from(a)
        vmin = self.level - self.window/2.0
        vmax = self.level + self.window/2.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            self._auto_wl_from(a)
            vmin = self.level - self.window/2.0
            vmax = self.level + self.window/2.0
        self.im.set_data(a)
        self.im.set_cmap(self._cmap_for_part())
        self.im.set_clim(vmin, vmax)
        self.ax_img.set_aspect(self._aspect_for_axis(self.axis))
        self.ax_img.set_title(f"{self.part} — {self.axis.upper()} slice {self.slice_idx+1}", fontsize=12)
        try:
            self.cbar.update_normal(self.im)
        except Exception:
            pass
        self.txt_info.set_text(f"W/L:\n{self.window:.6g} / {self.level:.6g}")
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
        # 重设 slice slider 的范围与标签
        self.slider_slice.valmin = 1
        self.slider_slice.valmax = self._max_slice()+1
        self.slider_slice.label.set_text(f"Slice ({self.axis.upper()})")
        # 跳到中间
        mid = (self._max_slice())//2
        self._set_slider_safely(self.slider_slice, mid+1)
        self.slice_idx = mid
        self._update_all(force=True)

    def _rotate(self, delta_deg):
        self.rot_deg = (self.rot_deg + delta_deg) % 360
        self._update_all()

    def _on_check(self, label):
        if label == "AutoWL":
            self.auto_wl = not self.auto_wl
            self._update_all()

    def _on_scroll(self, ev):
        if ev.inaxes != self.ax_img:
            return
        step = +1 if getattr(ev, "button", None) == "up" else -1
        self._step_slice(step)

    def _step_slice(self, step):
        new_idx = int(np.clip(self.slice_idx + step, 0, self._max_slice()))
        if new_idx != self.slice_idx:
            self.slice_idx = new_idx
            self._set_slider_safely(self.slider_slice, self.slice_idx+1)
            self._update_all()

    def _on_key(self, ev):
        if ev.key == "left":  self._step_slice(-1)
        elif ev.key == "right": self._step_slice(+1)
        elif ev.key == "x":   self.radio_axis.set_active(0)
        elif ev.key == "y":   self.radio_axis.set_active(1)
        elif ev.key == "z":   self.radio_axis.set_active(2)
        elif ev.key == "e":   self._rotate(+90)
        elif ev.key == "q":   self._rotate(-90)
        elif ev.key == "a":
            self.auto_wl = not self.auto_wl
            st = self.chk.get_status()[0]
            if st != self.auto_wl:
                self.chk.set_active(0)
            self._update_all()
        elif ev.key == "escape":
            plt.close(self.fig)

    def _on_press(self, ev):
        if ev.inaxes != self.ax_img:
            return
        if ev.button == 3:  # 右键开始 W/L 拖动
            self._wl_drag = (ev.x, ev.y, self.window, self.level)
            self._wl_dragging = True
            if self.cbar is not None:
                self.ax_cbar.set_visible(False)
                self._throttled_draw(force=True)
        elif ev.button == 1:  # 左键开始 slice/第4维 拖动
            d40 = self.extra_idx[0] if self.Nextra >= 1 else 0
            self._scroll_drag = (ev.x, ev.y, self.slice_idx, d40)

    def _on_release(self, ev):
        if self._wl_dragging and ev.button == 3:
            self._wl_drag = None
            self._wl_dragging = False
            if self.cbar is not None:
                self.ax_cbar.set_visible(True)
                self._throttled_draw(force=True)
        if self._scroll_drag is not None and ev.button == 1:
            self._scroll_drag = None

    def _on_move(self, ev):
        now = time.monotonic()
        if now - self._last_motion < 1.0/MOTION_FPS:
            return
        self._last_motion = now

        # 右键：W/L 拖动
        if self._wl_drag is not None:
            x0, y0, win0, lev0 = self._wl_drag
            dx = (ev.x - x0) if ev.x is not None else 0.0
            dy = (ev.y - y0) if ev.y is not None else 0.0
            rng = max(self.window, 1e-6)
            self.window = np.clip(win0 + dx * (rng/500.0), 1e-9, 1e12)
            self.level  = lev0 + dy * (rng/500.0)
            self._apply_wl()
            return

        # 左键：水平->第4维，垂直->slice
        if self._scroll_drag is not None and ev.inaxes == self.ax_img and ev.x is not None and ev.y is not None:
            x0, y0, s0, d40 = self._scroll_drag
            dx = ev.x - x0
            dy = ev.y - y0
            # 40px 作为1步
            step_t = int(np.round(dx / 40.0))
            step_s = int(np.round(-dy / 40.0))

            # 更新 slice
            new_slice = int(np.clip(s0 + step_s, 0, self._max_slice()))
            if new_slice != self.slice_idx:
                self.slice_idx = new_slice
                self._set_slider_safely(self.slider_slice, self.slice_idx+1)

            # 更新第4维（若存在）
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
    ap = argparse.ArgumentParser(description="BART CFL viewer (X/Y/Z, rotation, auto WL, extra dims)")
    ap.add_argument("file", nargs="?", help="base path WITHOUT extension (e.g. /path/to/imout)")
    ap.add_argument("--file", dest="file_flag", help="same as positional -- base path w/o extension")
    ap.add_argument("--vox", type=float, nargs=3, default=(1,1,1), help="voxel size dx dy dz")
    ap.add_argument("--title", default="", help="window title")
    ap.add_argument("--cmap", default="gray", help="matplotlib colormap")
    return ap.parse_args()

def main():
    args = parse_args()
    base = args.file_flag or args.file
    if not base:
        raise SystemExit("Please provide CFL base path as positional argument or via --file")
    vol = read_cfl(base)
    print("raw shape from .hdr:", vol.shape)
    viewer = Viewer(vol, vox=args.vox, title=args.title or (os.path.basename(base)+" (auto)"), cmap=args.cmap)
    plt.show()

if __name__ == "__main__":
    main()
