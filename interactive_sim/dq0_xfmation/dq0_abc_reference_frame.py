# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Circle, Arrow

# class DQ0Visualizer:
#     def __init__(self):
#         # Set up the figure with two subplots
#         self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
#         self.fig.suptitle('ABC Frame vs DQ0 Frame')
        
#         # Setup ABC frame (left plot)
#         self.ax1.set_xlim(-1.5, 1.5)
#         self.ax1.set_ylim(-1.5, 1.5)
#         self.ax1.grid(True)
#         self.ax1.set_aspect('equal')
#         self.ax1.set_title('ABC Frame (Stationary)')
        
#         # Setup DQ0 frame (right plot)
#         self.ax2.set_xlim(-1.5, 1.5)
#         self.ax2.set_ylim(-1.5, 1.5)
#         self.ax2.grid(True)
#         self.ax2.set_aspect('equal')
#         self.ax2.set_title('DQ0 Frame (Rotating)')
        
#         # Add reference circles
#         self.circle1 = self.ax1.add_patch(Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
#         self.circle2 = self.ax2.add_patch(Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
        
#         # Initialize arrows as empty lists to store the patches
#         self.abc_arrows = []
#         self.dq_arrows = []
        
#         # Add labels
#         self.ax1.text(1.1, 0, 'A', color='r')
#         self.ax1.text(-0.6, 0.966, 'B', color='g')
#         self.ax1.text(-0.6, -0.966, 'C', color='b')
        
#         self.ax2.text(1.1, 0, 'd', color='purple')
#         self.ax2.text(0, 1.1, 'q', color='orange')
    
#     def update(self, frame):
#         # Remove old arrows
#         for arrow in self.abc_arrows + self.dq_arrows:
#             arrow.remove()
#         self.abc_arrows.clear()
#         self.dq_arrows.clear()
        
#         # Update ABC frame (stationary)
#         self.abc_arrows.append(self.ax1.add_patch(Arrow(0, 0, 1, 0, color='r', width=0.1)))
#         self.abc_arrows.append(self.ax1.add_patch(Arrow(0, 0, -0.5, 0.866, color='g', width=0.1)))
#         self.abc_arrows.append(self.ax1.add_patch(Arrow(0, 0, -0.5, -0.866, color='b', width=0.1)))
        
#         # Update DQ frame (rotating)
#         angle = frame * np.pi / 30  # Rotation angle
        
#         # Calculate rotated coordinates for d-axis
#         dx = np.cos(angle)
#         dy = np.sin(angle)
#         self.dq_arrows.append(self.ax2.add_patch(Arrow(0, 0, dx, dy, color='purple', width=0.1)))
        
#         # Calculate rotated coordinates for q-axis (90 degrees ahead)
#         qx = -np.sin(angle)
#         qy = np.cos(angle)
#         self.dq_arrows.append(self.ax2.add_patch(Arrow(0, 0, qx, qy, color='orange', width=0.1)))
        
#         return self.abc_arrows + self.dq_arrows + [self.circle1, self.circle2]

# # Create and run animation
# visualizer = DQ0Visualizer()
# anim = FuncAnimation(visualizer.fig, visualizer.update, frames=60, 
#                     interval=50, blit=True, repeat=True)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arrow
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DQ0Visualizer:
    def __init__(self):
        # Set up the figure with three subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(2, 2)
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])  # ABC frame
        self.ax2 = self.fig.add_subplot(self.gs[0, 1])  # DQ frame
        self.ax3 = self.fig.add_subplot(self.gs[1, :])  # Voltage waveforms
        
        self.fig.suptitle('ABC Frame vs DQ0 Frame with Phase Voltages')
        
        # Setup ABC frame (top left plot)
        self.ax1.set_xlim(-1.5, 1.5)
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.grid(True)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('ABC Frame (Stationary)')
        
        # Setup DQ0 frame (top right plot)
        self.ax2.set_xlim(-1.5, 1.5)
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.grid(True)
        self.ax2.set_aspect('equal')
        self.ax2.set_title('DQ0 Frame (Rotating)')
        
        # Setup voltage waveforms plot (bottom)
        self.ax3.set_xlim(0, 2*np.pi)
        self.ax3.set_ylim(-1.5, 1.5)
        self.ax3.grid(True)
        self.ax3.set_title('Phase Voltages')
        self.ax3.set_xlabel('Angle (rad)')
        self.ax3.set_ylabel('Voltage (pu)')
        
        # Add reference circles
        self.circle1 = self.ax1.add_patch(Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
        self.circle2 = self.ax2.add_patch(Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
        
        # Initialize arrows and lines
        self.abc_arrows = []
        self.dq_arrows = []
        
        # Initialize voltage waveform lines
        x = np.linspace(0, 2*np.pi, 100)
        self.line_a, = self.ax3.plot([], [], 'r-', label='Va')
        self.line_b, = self.ax3.plot([], [], 'g-', label='Vb')
        self.line_c, = self.ax3.plot([], [], 'b-', label='Vc')
        self.time_marker, = self.ax3.plot([], [], 'k.', markersize=10)
        self.ax3.legend()
        
        # Add labels
        self.ax1.text(1.1, 0, 'A', color='r')
        self.ax1.text(-0.6, 0.966, 'B', color='g')
        self.ax1.text(-0.6, -0.966, 'C', color='b')
        
        self.ax2.text(1.1, 0, 'd', color='purple')
        self.ax2.text(0, 1.1, 'q', color='orange')
        
        # Store time points for voltage waveforms
        self.t = np.linspace(0, 2*np.pi, 100)
        
    def update(self, frame):
        # Remove old arrows
        for arrow in self.abc_arrows + self.dq_arrows:
            arrow.remove()
        self.abc_arrows.clear()
        self.dq_arrows.clear()
        
        angle = frame * np.pi / 30  # Current angle
        
        # Calculate instantaneous voltages
        va = np.cos(angle)
        vb = np.cos(angle - 2*np.pi/3)
        vc = np.cos(angle + 2*np.pi/3)
        
        # Update ABC frame with varying arrow lengths
        self.abc_arrows.append(self.ax1.add_patch(Arrow(0, 0, va, 0, color='r', width=0.1)))
        self.abc_arrows.append(self.ax1.add_patch(
            Arrow(0, 0, -0.5*vb, 0.866*vb, color='g', width=0.1)))
        self.abc_arrows.append(self.ax1.add_patch(
            Arrow(0, 0, -0.5*vc, -0.866*vc, color='b', width=0.1)))
        
        # Update DQ frame (rotating)
        dx = np.cos(angle)
        dy = np.sin(angle)
        self.dq_arrows.append(self.ax2.add_patch(Arrow(0, 0, dx, dy, color='purple', width=0.1)))
        
        qx = -np.sin(angle)
        qy = np.cos(angle)
        self.dq_arrows.append(self.ax2.add_patch(Arrow(0, 0, qx, qy, color='orange', width=0.1)))
        
        # Update voltage waveforms
        va_wave = np.cos(self.t)
        vb_wave = np.cos(self.t - 2*np.pi/3)
        vc_wave = np.cos(self.t + 2*np.pi/3)
        
        self.line_a.set_data(self.t, va_wave)
        self.line_b.set_data(self.t, vb_wave)
        self.line_c.set_data(self.t, vc_wave)
        self.time_marker.set_data([angle % (2*np.pi)], [np.cos(angle)])
        
        return self.abc_arrows + self.dq_arrows + [self.circle1, self.circle2, 
                self.line_a, self.line_b, self.line_c, self.time_marker]

# Create and run animation
visualizer = DQ0Visualizer()
anim = FuncAnimation(visualizer.fig, visualizer.update, frames=60, 
                    interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()