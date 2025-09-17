import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

class PowerAngleVisualizer:
    def __init__(self):
        # Set up the figure with subplots
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        
        # Power angle curve subplot
        self.ax1 = plt.subplot(gs[0, :])
        
        # Generator animation subplot
        self.ax2 = plt.subplot(gs[1, 0])
        self.ax2.set_aspect('equal')
        
        # Operating point info subplot
        self.ax3 = plt.subplot(gs[1, 1])
        
        # Initialize plot data
        self.delta = np.linspace(0, 2*np.pi, 1000)
        self.Pe = np.sin(self.delta)
        
        # Set up power-angle curve
        self.ax1.plot(np.degrees(self.delta), self.Pe, 'b-', label='Power-Angle Curve')
        self.ax1.grid(True)
        self.ax1.set_xlabel('Rotor Angle δ (degrees)')
        self.ax1.set_ylabel('Electrical Power (Pe)')
        self.ax1.set_title('Power-Angle Relationship')
        
        # Add vertical lines for stability regions
        self.ax1.axvline(x=90, color='r', linestyle='--', alpha=0.5, label='Stability Limit')
        self.ax1.axvline(x=270, color='r', linestyle='--', alpha=0.5)
        
        # Add region labels
        self.ax1.text(45, 0.5, 'Stable\nRegion', horizontalalignment='center')
        self.ax1.text(135, -0.5, 'Unstable\nRegion', horizontalalignment='center')
        
        # Set up moving point on curve
        self.point, = self.ax1.plot([], [], 'ro')
        
        # Set up generator visualization
        self.rotor, = self.ax2.plot([], [], 'b-', linewidth=3, label='Rotor')
        self.stator, = self.ax2.plot([], [], 'g-', linewidth=3, label='Stator')
        self.ax2.set_xlim(-1.5, 1.5)
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.grid(True)
        self.ax2.set_title('Generator Visualization')
        
        # Set up operating point info display
        self.ax3.axis('off')
        self.info_text = self.ax3.text(0.1, 0.5, '', fontsize=10)
        
        # Add overall title
        self.fig.suptitle('Power-Angle Relationship', fontsize=16)
        
        # Add legends
        self.ax1.legend()
        self.ax2.legend()

    def update(self, frame):
        # Update point position on power-angle curve
        angle_deg = frame
        angle_rad = np.radians(angle_deg)
        power = np.sin(angle_rad)
        
        self.point.set_data([angle_deg], [power])
        
        # Update generator visualization
        rotor_x = np.cos(np.linspace(0, 2*np.pi, 100))
        rotor_y = np.sin(np.linspace(0, 2*np.pi, 100))
        self.rotor.set_data(rotor_x, rotor_y)
        
        # Add rotor line
        rotor_line_x = [0, np.cos(angle_rad)]
        rotor_line_y = [0, np.sin(angle_rad)]
        self.rotor.set_data(rotor_line_x, rotor_line_y)
        
        # Add stator line (always at 0 degrees)
        stator_line_x = [0, 1]
        stator_line_y = [0, 0]
        self.stator.set_data(stator_line_x, stator_line_y)
        
        # Update information display
        stability = "Stable" if 0 <= angle_deg <= 90 else "Unstable"
        info = f"Rotor Angle (δ): {angle_deg:.1f}°\n"
        info += f"Electric Power (Pe): {power:.3f} pu\n"
        info += f"Operating Region: {stability}\n"
        info += f"Max Power: {1.0:.2f} pu at δ = 90°"
        self.info_text.set_text(info)
        
        return self.point, self.rotor, self.stator, self.info_text

    def animate(self):
        anim = FuncAnimation(self.fig, self.update, 
                           frames=np.linspace(0, 360, 360),
                           interval=50, blit=True, repeat=True)
        plt.show()

# Create and run animation
visualizer = PowerAngleVisualizer()
visualizer.animate()