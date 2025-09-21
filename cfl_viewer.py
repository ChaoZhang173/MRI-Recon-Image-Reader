#!/usr/bin/env python3
"""
Advanced CFL Viewer - Multi-directional slicing, rotation, and multi-dimensional data navigation

Features:
- Multi-directional slicing (X/Y/Z)
- 90-degree rotation
- Multi-dimensional data navigation
- Automatic dimension processing
- Perfect layout with no overlapping buttons
- Auto window/level adjustment

Usage:
    python cfl_viewer.py <cfl_file_path>
    or modify the DEFAULT_FILE_PATH in the script

Author: MRI Research Group
License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Default file path - modify this if you want to hardcode a file
DEFAULT_FILE_PATH = None

try:
    from cfl_reader import readcfl
except ImportError:
    print("Error: cfl_reader module not found")
    print("Please ensure cfl_reader.py is in the same directory")
    sys.exit(1)

class AdvancedCFLViewer:
    """Advanced CFL data viewer with multi-dimensional navigation capabilities"""
    
    def __init__(self, data, title="Advanced CFL Viewer"):
        """
        Initialize the CFL viewer
        
        Parameters:
        -----------
        data : numpy.ndarray
            Complex-valued CFL data
        title : str
            Window title
        """
        # Process data
        self.original_data = data
        print(f"Original data shape: {data.shape}")
        
        # Process dimensions to handle multi-dimensional data
        self.processed_data = self.process_data_dimensions(data)
        print(f"Processed data shape: {self.processed_data.shape}")
        
        self.title = title
        
        # Initialize slice direction and state
        self.slice_direction = 'z'  # 'x', 'y', 'z'
        self.slice_directions = ['x', 'y', 'z']
        self.rotation_angle = 0  # 0, 90, 180, 270
        
        # Setup dimensions based on processed data
        self.setup_dimensions()
        
        # Initialize state
        self.current_slice = self.get_middle_slice()
        self.current_mode = 'magnitude'
        self.current_cmap = 'gray'
        
        # Multi-dimensional data current indices
        self.current_dim1_idx = 0 if self.dim1_size > 1 else 0
        self.current_dim2_idx = 0 if self.dim2_size > 1 else 0
        
        # Display modes and colormaps
        self.modes = ['magnitude', 'phase', 'real', 'imaginary']
        self.cmaps = ['gray', 'jet', 'viridis', 'hot', 'cool']
        
        # Window/Level for auto adjustment
        self.auto_wl = True
        self.window = 1.0
        self.level = 0.5
        
        # Initialize display components
        self.fig = None
        self.ax_image = None
        self.im = None
        self.cbar = None
        
        self.setup_advanced_layout()
        self.connect_events()
        
        # Initial display
        self.update_display()
        
    def process_data_dimensions(self, data):
        """
        Process data dimensions: keep first 3 dimensions, handle others intelligently
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data array
            
        Returns:
        --------
        numpy.ndarray
            Processed data with manageable dimensions
        """
        if data.ndim <= 3:
            return data
            
        # Keep first 3 dimensions
        shape = list(data.shape)
        front_3 = shape[:3]
        remaining = shape[3:]
        
        # Find non-unit dimensions in remaining dimensions
        non_one_dims = [i for i, dim in enumerate(remaining) if dim != 1]
        
        if len(non_one_dims) == 0:
            # All remaining dimensions are 1, extract first 3 dimensions
            indices = [slice(None)] * 3 + [0] * (data.ndim - 3)
            return data[tuple(indices)]
        elif len(non_one_dims) <= 2:
            # At most 2 non-unit dimensions, keep them
            indices = [slice(None)] * 3  # Keep first 3 dimensions unchanged
            
            # For remaining dimensions, keep non-unit ones, set others to 0
            kept_dims = 0
            for i, dim in enumerate(remaining):
                if dim != 1 and kept_dims < 2:
                    indices.append(slice(None))
                    kept_dims += 1
                else:
                    indices.append(0)
            
            return data[tuple(indices)]
        else:
            # More than 2 non-unit dimensions, keep only first 2
            print("Warning: More than 2 non-unit dimensions found, using first 2")
            indices = [slice(None)] * 3
            
            kept_dims = 0
            for i, dim in enumerate(remaining):
                if dim != 1 and kept_dims < 2:
                    indices.append(slice(None))
                    kept_dims += 1
                else:
                    indices.append(0)
                    
            return data[tuple(indices)]
    
    def setup_dimensions(self):
        """Setup dimension information from processed data"""
        shape = self.processed_data.shape
        
        # First 3 dimensions (x, y, z)
        self.nx, self.ny, self.nz = shape[0], shape[1], shape[2]
        
        # Additional dimensions
        if len(shape) > 3:
            self.dim1_size = shape[3] if len(shape) > 3 else 1
            self.dim2_size = shape[4] if len(shape) > 4 else 1
        else:
            self.dim1_size = 1
            self.dim2_size = 1
            
        print(f"Dimensions: x={self.nx}, y={self.ny}, z={self.nz}")
        if self.dim1_size > 1:
            print(f"Extra dimension 1: {self.dim1_size}")
        if self.dim2_size > 1:
            print(f"Extra dimension 2: {self.dim2_size}")
    
    def get_middle_slice(self):
        """Get middle slice for current slice direction"""
        if self.slice_direction == 'x':
            return self.nx // 2
        elif self.slice_direction == 'y':
            return self.ny // 2
        else:  # 'z'
            return self.nz // 2
            
    def get_max_slice(self):
        """Get maximum slice number for current slice direction"""
        if self.slice_direction == 'x':
            return self.nx - 1
        elif self.slice_direction == 'y':
            return self.ny - 1
        else:  # 'z'
            return self.nz - 1
        
    def setup_advanced_layout(self):
        """Create advanced layout with all new features"""
        # Create larger figure
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.suptitle(self.title, fontsize=16, fontweight='bold')
        
        # Main image area
        self.ax_image = plt.axes([0.05, 0.25, 0.6, 0.65])
        self.ax_image.set_aspect('equal')
        
        # Slice control slider
        ax_slice = plt.axes([0.05, 0.15, 0.6, 0.04])
        self.slider_slice = Slider(
            ax_slice, f'Slice ({self.slice_direction.upper()})', 0, self.get_max_slice(), 
            valinit=self.current_slice, valfmt='%d'
        )
        
        # Multi-dimensional data control sliders
        if self.dim1_size > 1:
            ax_dim1 = plt.axes([0.05, 0.08, 0.6, 0.04])
            self.slider_dim1 = Slider(
                ax_dim1, 'Dimension 1', 0, self.dim1_size - 1, 
                valinit=self.current_dim1_idx, valfmt='%d'
            )
        else:
            self.slider_dim1 = None
            
        if self.dim2_size > 1:
            ax_dim2 = plt.axes([0.05, 0.01, 0.6, 0.04])
            self.slider_dim2 = Slider(
                ax_dim2, 'Dimension 2', 0, self.dim2_size - 1, 
                valinit=self.current_dim2_idx, valfmt='%d'
            )
        else:
            self.slider_dim2 = None
        
        # Right side control panel
        self.setup_right_control_panel()
        
        print("Advanced layout with multi-dimensional controls created")
        
    def setup_right_control_panel(self):
        """Setup right side control panel"""
        right_start = 0.68
        panel_width = 0.3
        
        # 1. Slice direction selection
        ax_slice_dir = plt.axes([right_start, 0.8, 0.25, 0.12])
        ax_slice_dir.set_title("Slice Direction", fontsize=12, fontweight='bold')
        self.radio_slice_dir = RadioButtons(ax_slice_dir, self.slice_directions)
        self.radio_slice_dir.set_active(2)  # Default z direction
        
        # 2. Display mode selection
        ax_mode = plt.axes([right_start, 0.65, 0.25, 0.13])
        ax_mode.set_title("Display Mode", fontsize=12, fontweight='bold')
        self.radio_mode = RadioButtons(ax_mode, self.modes)
        self.radio_mode.set_active(0)
        
        # 3. Colormap selection
        ax_cmap = plt.axes([right_start, 0.48, 0.25, 0.15])
        ax_cmap.set_title("Colormap", fontsize=12, fontweight='bold')
        self.radio_cmap = RadioButtons(ax_cmap, self.cmaps)
        self.radio_cmap.set_active(0)
        
        # 4. Rotation control buttons
        button_width = 0.12
        button_height = 0.05
        
        # Rotation buttons
        ax_rotate_cw = plt.axes([right_start, 0.38, button_width, button_height])
        self.btn_rotate_cw = Button(ax_rotate_cw, 'Rotate CW')
        
        ax_rotate_ccw = plt.axes([right_start + 0.13, 0.38, button_width, button_height])
        self.btn_rotate_ccw = Button(ax_rotate_ccw, 'Rotate CCW')
        
        # 5. Control buttons
        ax_auto_wl = plt.axes([right_start, 0.31, 0.25, button_height])
        self.btn_auto_wl = Button(ax_auto_wl, 'Auto W/L')
        
        ax_reset = plt.axes([right_start, 0.24, 0.25, button_height])
        self.btn_reset = Button(ax_reset, 'Reset')
        
        # Navigation buttons
        ax_prev = plt.axes([right_start, 0.17, button_width, button_height])
        self.btn_prev = Button(ax_prev, 'Prev')
        
        ax_next = plt.axes([right_start + 0.13, 0.17, button_width, button_height])
        self.btn_next = Button(ax_next, 'Next')
        
        # 6. Information display area
        ax_info = plt.axes([right_start, 0.02, 0.28, 0.13])
        ax_info.set_title("Information", fontsize=12, fontweight='bold')
        ax_info.axis('off')
        self.ax_info = ax_info
        
    def connect_events(self):
        """Connect event handlers"""
        # Slider events
        self.slider_slice.on_changed(self.on_slice_change)
        if self.slider_dim1:
            self.slider_dim1.on_changed(self.on_dim1_change)
        if self.slider_dim2:
            self.slider_dim2.on_changed(self.on_dim2_change)
            
        # RadioButton events
        self.radio_slice_dir.on_clicked(self.on_slice_direction_change)
        self.radio_mode.on_clicked(self.on_mode_change)
        self.radio_cmap.on_clicked(self.on_cmap_change)
        
        # Button events
        self.btn_rotate_cw.on_clicked(self.on_rotate_cw)
        self.btn_rotate_ccw.on_clicked(self.on_rotate_ccw)
        self.btn_prev.on_clicked(self.on_prev_slice)
        self.btn_next.on_clicked(self.on_next_slice)
        self.btn_auto_wl.on_clicked(self.on_auto_wl)
        self.btn_reset.on_clicked(self.on_reset)
        
        # Mouse and keyboard events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        print("Advanced controls connected")
        
    def get_current_slice_data(self):
        """Get 2D data based on slice direction and current indices"""
        # Build indices
        if len(self.processed_data.shape) == 3:
            # 3D data
            if self.slice_direction == 'x':
                slice_data = self.processed_data[self.current_slice, :, :]
            elif self.slice_direction == 'y':
                slice_data = self.processed_data[:, self.current_slice, :]
            else:  # 'z'
                slice_data = self.processed_data[:, :, self.current_slice]
        elif len(self.processed_data.shape) == 4:
            # 4D data
            if self.slice_direction == 'x':
                slice_data = self.processed_data[self.current_slice, :, :, self.current_dim1_idx]
            elif self.slice_direction == 'y':
                slice_data = self.processed_data[:, self.current_slice, :, self.current_dim1_idx]
            else:  # 'z'
                slice_data = self.processed_data[:, :, self.current_slice, self.current_dim1_idx]
        elif len(self.processed_data.shape) == 5:
            # 5D data
            if self.slice_direction == 'x':
                slice_data = self.processed_data[self.current_slice, :, :, self.current_dim1_idx, self.current_dim2_idx]
            elif self.slice_direction == 'y':
                slice_data = self.processed_data[:, self.current_slice, :, self.current_dim1_idx, self.current_dim2_idx]
            else:  # 'z'
                slice_data = self.processed_data[:, :, self.current_slice, self.current_dim1_idx, self.current_dim2_idx]
        else:
            # Default case
            slice_data = self.processed_data[:, :, self.current_slice] if self.processed_data.ndim >= 3 else self.processed_data
            
        return slice_data
    
    def apply_rotation(self, data):
        """Apply rotation to data"""
        if self.rotation_angle == 0:
            return data
        elif self.rotation_angle == 90:
            return np.rot90(data, k=1)
        elif self.rotation_angle == 180:
            return np.rot90(data, k=2)
        elif self.rotation_angle == 270:
            return np.rot90(data, k=3)
        else:
            return data
    
    def get_display_data(self):
        """Get current display data"""
        # Get slice data
        slice_data = self.get_current_slice_data()
        
        # Apply rotation
        slice_data = self.apply_rotation(slice_data)
            
        # Apply display mode
        if self.current_mode == 'magnitude':
            return np.abs(slice_data)
        elif self.current_mode == 'phase':
            return np.angle(slice_data)
        elif self.current_mode == 'real':
            return np.real(slice_data)
        elif self.current_mode == 'imaginary':
            return np.imag(slice_data)
        else:
            return np.abs(slice_data)
    
    def update_display(self):
        """Update image display"""
        # Get current data
        display_data = self.get_display_data()
        
        # Apply auto window/level
        if self.auto_wl:
            self.apply_auto_wl(display_data)
        
        # Select colormap
        cmap = 'hsv' if self.current_mode == 'phase' else self.current_cmap
        
        # Update or create image
        if hasattr(self, 'im') and self.im is not None:
            # Update existing image
            self.im.set_array(display_data)
            self.im.set_cmap(cmap)
            self.im.set_clim(vmin=self.level - self.window/2, 
                            vmax=self.level + self.window/2)
            
            # Update extent to match new data shape
            h, w = display_data.shape
            self.im.set_extent([-0.5, w-0.5, -0.5, h-0.5])
        else:
            # First time creating image
            self.im = self.ax_image.imshow(
                display_data, 
                cmap=cmap, 
                origin='lower',
                aspect='equal',
                vmin=self.level - self.window/2,
                vmax=self.level + self.window/2,
                interpolation='nearest'
            )
            
        # Set axis limits
        h, w = display_data.shape
        self.ax_image.set_xlim(-0.5, w-0.5)
        self.ax_image.set_ylim(-0.5, h-0.5)
        self.ax_image.set_autoscale_on(False)
        
        # Update title
        title = f'{self.current_mode.title()} - {self.slice_direction.upper()}-slice {self.current_slice+1}'
        if self.dim1_size > 1:
            title += f' - Dim1: {self.current_dim1_idx+1}'
        if self.dim2_size > 1:
            title += f' - Dim2: {self.current_dim2_idx+1}'
        if self.rotation_angle != 0:
            title += f' - Rot: {self.rotation_angle}deg'
            
        self.ax_image.set_title(title, fontsize=14, pad=15)
        
        # Update colorbar
        self.update_colorbar_safe()
        
        # Update information display
        self.update_info_display(display_data)
        
        # Refresh display
        self.fig.canvas.draw_idle()
        
    def update_colorbar_safe(self):
        """Safely update colorbar"""
        try:
            if hasattr(self, 'cbar') and self.cbar is not None:
                self.cbar.update_normal(self.im)
            else:
                self.cbar = self.fig.colorbar(
                    self.im, ax=self.ax_image, 
                    fraction=0.04, pad=0.02, shrink=0.8
                )
        except Exception:
            pass
    
    def update_info_display(self, data):
        """Update information display"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"""Shape: {data.shape}
Direction: {self.slice_direction.upper()}
Slice: {self.current_slice+1}
Mode: {self.current_mode}
Rotation: {self.rotation_angle}deg
Window: {self.window:.4f}
Level: {self.level:.4f}
Range: [{np.min(data):.4f}, {np.max(data):.4f}]"""

        if self.dim1_size > 1:
            info_text += f"\nDim1: {self.current_dim1_idx+1}/{self.dim1_size}"
        if self.dim2_size > 1:
            info_text += f"\nDim2: {self.current_dim2_idx+1}/{self.dim2_size}"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    def apply_auto_wl(self, data):
        """Apply automatic window/level adjustment"""
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        
        if data_max == data_min:
            data_max = data_min + 1
            
        self.window = data_max - data_min
        self.level = (data_max + data_min) / 2
        
        print(f"Auto W/L: Window={self.window:.4f}, Level={self.level:.4f}")
    
    # Event handlers
    def on_slice_change(self, val):
        """Handle slice slider change"""
        self.current_slice = int(val)
        self.update_display()
        
    def on_dim1_change(self, val):
        """Handle dimension 1 slider change"""
        self.current_dim1_idx = int(val)
        self.update_display()
        
    def on_dim2_change(self, val):
        """Handle dimension 2 slider change"""
        self.current_dim2_idx = int(val)
        self.update_display()
        
    def on_slice_direction_change(self, label):
        """Switch slice direction"""
        self.slice_direction = label
        # Reset slider
        max_slice = self.get_max_slice()
        self.current_slice = self.get_middle_slice()
        
        # Update slider
        self.slider_slice.valmax = max_slice
        self.slider_slice.set_val(self.current_slice)
        self.slider_slice.label.set_text(f'Slice ({label.upper()})')
        
        print(f"Switched to {label.upper()}-direction slicing")
        self.update_display()
        
    def on_mode_change(self, label):
        """Handle display mode change"""
        self.current_mode = label
        self.update_display()
        
    def on_cmap_change(self, label):
        """Handle colormap change"""
        self.current_cmap = label
        self.update_display()
        
    def on_rotate_cw(self, event):
        """Rotate clockwise 90 degrees"""
        self.rotation_angle = (self.rotation_angle + 90) % 360
        print(f"Rotated clockwise to {self.rotation_angle} degrees")
        self.update_display()
        
    def on_rotate_ccw(self, event):
        """Rotate counter-clockwise 90 degrees"""
        self.rotation_angle = (self.rotation_angle - 90) % 360
        print(f"Rotated counter-clockwise to {self.rotation_angle} degrees")
        self.update_display()
        
    def on_prev_slice(self, event):
        """Go to previous slice"""
        if self.current_slice > 0:
            self.current_slice -= 1
            self.slider_slice.set_val(self.current_slice)
            
    def on_next_slice(self, event):
        """Go to next slice"""
        max_slice = self.get_max_slice()
        if self.current_slice < max_slice:
            self.current_slice += 1
            self.slider_slice.set_val(self.current_slice)
            
    def on_auto_wl(self, event):
        """Auto window/level button callback"""
        self.auto_wl = True
        self.update_display()
        
    def on_reset(self, event):
        """Reset view to defaults"""
        self.current_slice = self.get_middle_slice()
        self.current_mode = 'magnitude'
        self.current_cmap = 'gray'
        self.rotation_angle = 0
        self.auto_wl = True
        self.current_dim1_idx = 0
        self.current_dim2_idx = 0
        
        # Reset UI controls
        self.slider_slice.set_val(self.current_slice)
        if self.slider_dim1:
            self.slider_dim1.set_val(0)
        if self.slider_dim2:
            self.slider_dim2.set_val(0)
        self.radio_mode.set_active(0)
        self.radio_cmap.set_active(0)
        
        print("All settings reset to defaults")
        self.update_display()
        
    def on_mouse_click(self, event):
        """Handle mouse click events"""
        if event.inaxes == self.ax_image and event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            display_data = self.get_display_data()
            if 0 <= x < display_data.shape[1] and 0 <= y < display_data.shape[0]:
                value = display_data[y, x]
                print(f"Pixel ({x}, {y}): {value:.6f}")
                
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'left' and self.current_slice > 0:
            self.current_slice -= 1
            self.slider_slice.set_val(self.current_slice)
        elif event.key == 'right' and self.current_slice < self.get_max_slice():
            self.current_slice += 1
            self.slider_slice.set_val(self.current_slice)
        elif event.key == 'a':
            self.on_auto_wl(None)
        elif event.key == 'r':
            self.on_rotate_cw(None)
        elif event.key == 'x':
            self.radio_slice_dir.set_active(0)
        elif event.key == 'y':
            self.radio_slice_dir.set_active(1)
        elif event.key == 'z':
            self.radio_slice_dir.set_active(2)
            
    def on_scroll(self, event):
        """Handle mouse scroll events"""
        if event.inaxes == self.ax_image:
            if event.button == 'up' and self.current_slice < self.get_max_slice():
                self.current_slice += 1
                self.slider_slice.set_val(self.current_slice)
            elif event.button == 'down' and self.current_slice > 0:
                self.current_slice -= 1
                self.slider_slice.set_val(self.current_slice)

def main():
    """Main function"""
    # Check for command line argument or use default
    if len(sys.argv) == 2:
        cfl_file = sys.argv[1]
    elif DEFAULT_FILE_PATH is not None:
        cfl_file = DEFAULT_FILE_PATH
        print(f"Using default file path: {cfl_file}")
    else:
        print("Usage: python cfl_viewer.py <cfl_file>")
        print("Or set DEFAULT_FILE_PATH in the script")
        return
    
    print("Advanced CFL Viewer")
    print("=" * 50)
    print("Features:")
    print("  - Multi-directional slicing (X/Y/Z)")
    print("  - 90-degree rotation")
    print("  - Multi-dimensional data navigation")
    print("  - Automatic dimension processing")
    print("=" * 50)
    
    try:
        print(f"Using {matplotlib.get_backend()} backend")
        
        # Remove .cfl extension if present
        if cfl_file.endswith('.cfl'):
            filename_base = cfl_file[:-4]
        else:
            filename_base = cfl_file
            
        print(f"Loading: {filename_base}")
        data = readcfl(filename_base)
        print("Loaded successfully")
        
        print(f"Data: {data.shape}, {data.dtype}")
        data_size_mb = data.nbytes / (1024**2)
        print(f"Memory: {data_size_mb:.1f} MB")
        
        print("\nStarting advanced viewer...")
        
        viewer = AdvancedCFLViewer(data, "Advanced Multi-Dimensional CFL Viewer")
        plt.show()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Viewer error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Session ended")

if __name__ == '__main__':
    main()
