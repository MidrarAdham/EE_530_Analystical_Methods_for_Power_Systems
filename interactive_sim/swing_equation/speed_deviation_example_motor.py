import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

class MotorVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])
        
        # Speed plot
        self.ax1 = plt.subplot(gs[0])
        # Delta plot
        self.ax2 = plt.subplot(gs[1])
        # Information display
        self.ax3 = plt.subplot(gs[2])
        
        # Time array
        self.t = np.linspace(0, 10, 1000)
        
        # System parameters
        self.ws = 377  # rad/s (60 Hz)
        self.H = 10.0   # Inertia constant
        
        # Initialize plots
        self.w_line, = self.ax1.plot([], [], 'b-', label='Actual Speed (ω)')
        self.ws_line, = self.ax1.plot([], [], 'r--', label='No-load Speed')
        self.delta_line, = self.ax2.plot([], [], 'g-', label='Load Angle (δ)')
        
        # Set up axes
        self.setup_axes()
        
        # Information text
        self.ax3.axis('off')
        self.info_text = self.ax3.text(0.1, 0.5, '', fontsize=10)
        
        # Add disturbance
        self.disturbance_time = 2.0  # Time of disturbance
        self.load_change = 0.05      # 10% increase in mechanical load

    def setup_axes(self):
        # Speed plot setup
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(375, 379)
        self.ax1.grid(True)
        self.ax1.set_ylabel('Angular Speed (rad/s)')
        self.ax1.set_title('Motor Speed Response')
        self.ax1.legend()
        
        # Delta plot setup
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(-10, 40)
        self.ax2.grid(True)
        self.ax2.set_ylabel('Load Angle (degrees)')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.legend()
        
        # Overall title
        self.fig.suptitle('Motor Response to Load Increase', fontsize=16)

    def calculate_response(self, t_current):
        # Initial conditions
        w0 = self.ws
        delta0 = 5  # Initial load angle (degrees)
        
        # Create arrays for w and delta
        w = np.zeros_like(t_current)
        delta = np.zeros_like(t_current)
        
        # Calculate response
        mask_before = t_current < self.disturbance_time
        mask_after = t_current >= self.disturbance_time
        
        # Before disturbance
        w[mask_before] = w0
        delta[mask_before] = delta0
        
        # After disturbance - note the negative sign for load increase
        t_adj = t_current[mask_after] - self.disturbance_time
        w[mask_after] = w0 - self.load_change * self.ws / (2 * self.H) * \
            np.sin(np.sqrt(self.ws / (2 * self.H)) * t_adj)
        
        delta[mask_after] = delta0 + self.load_change * self.ws / (2 * self.H) * \
            (1 - np.cos(np.sqrt(self.ws / (2 * self.H)) * t_adj))
        
        return w, delta

    def update(self, frame):
        # Calculate time points up to current frame
        t_current = self.t[:frame]
        
        if len(t_current) > 0:
            # Calculate response
            w, delta = self.calculate_response(t_current)
            
            # Update speed plot
            self.w_line.set_data(t_current, w)
            self.ws_line.set_data(t_current, np.ones_like(t_current) * self.ws)
            
            # Update delta plot
            self.delta_line.set_data(t_current, delta)
            
            # Update information display
            info = f"Time: {t_current[-1]:.2f} s\n"
            info += f"Speed Deviation: {(w[-1] - self.ws):.3f} rad/s\n"
            info += f"Load Angle: {delta[-1]:.2f}°\n"
            info += f"Inertia Constant (H): {self.H} s"
            if t_current[-1] >= self.disturbance_time:
                info += f"\nDisturbance: +{self.load_change*100}% Load increase"
            self.info_text.set_text(info)
        
        return self.w_line, self.ws_line, self.delta_line, self.info_text

    def animate(self):
        anim = FuncAnimation(self.fig, self.update,
                           frames=len(self.t),
                           interval=20, blit=True)
        plt.show()

# Create and run animation
visualizer = MotorVisualizer()
visualizer.animate()