import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arrow

class UnwrappingVisualizer:
    def __init__(self):
        # Create two subplots side by side
        self.fig, (self.ax_rotate, self.ax_unwrap) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle('Unwrapping Rotating Vector to DQ Components', fontsize=14)
        
        # Setup rotating vector plot
        self.ax_rotate.set_xlim(-1.5, 1.5)
        self.ax_rotate.set_ylim(-1.5, 1.5)
        self.ax_rotate.grid(True)
        self.ax_rotate.set_aspect('equal')
        self.ax_rotate.set_title('Rotating Space')
        
        # Setup unwrapped plot
        self.ax_unwrap.set_xlim(0, 4*np.pi)
        self.ax_unwrap.set_ylim(-1.5, 1.5)
        self.ax_unwrap.grid(True)
        self.ax_unwrap.set_title('Unwrapped DQ Components')
        self.ax_unwrap.set_xlabel('Angle (rad)')
        self.ax_unwrap.set_ylabel('Magnitude')
        
        # Add reference circle
        self.circle = self.ax_rotate.add_patch(Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
        
        # Initialize vectors and lines
        self.rotating_vectors = []
        self.unwrapped_lines_d = []
        self.unwrapped_lines_q = []
        self.time_points = np.linspace(0, 4*np.pi, 100)
        
        # Add legends
        self.ax_unwrap.plot([], [], 'purple', label='d-component')
        self.ax_unwrap.plot([], [], 'orange', label='q-component')
        self.ax_unwrap.legend()
        
    def update(self, frame):
        # Clear previous vectors and lines
        for vector in self.rotating_vectors:
            vector.remove()
        self.rotating_vectors.clear()
        
        for line in self.unwrapped_lines_d + self.unwrapped_lines_q:
            line.remove()
        self.unwrapped_lines_d.clear()
        self.unwrapped_lines_q.clear()
        
        theta = frame * np.pi / 30
        
        # Calculate vector components
        vd = np.cos(0)  # Constant in dq frame
        vq = np.sin(0)  # Constant in dq frame
        
        # Rotating vector
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Draw rotating vector
        self.rotating_vectors.append(self.ax_rotate.add_patch(
            Arrow(0, 0, x, y, color='red', width=0.1)))
        
        # Draw d-axis
        self.rotating_vectors.append(self.ax_rotate.add_patch(
            Arrow(0, 0, np.cos(theta), np.sin(theta), color='purple', width=0.05)))
        
        # Draw q-axis
        self.rotating_vectors.append(self.ax_rotate.add_patch(
            Arrow(0, 0, -np.sin(theta), np.cos(theta), color='orange', width=0.05)))
        
        # Calculate unwrapped components
        d_component = np.cos(self.time_points - theta)
        q_component = np.sin(self.time_points - theta)
        
        # Plot unwrapped components
        self.unwrapped_lines_d.append(self.ax_unwrap.plot(
            self.time_points, d_component, 'purple', alpha=0.5)[0])
        self.unwrapped_lines_q.append(self.ax_unwrap.plot(
            self.time_points, q_component, 'orange', alpha=0.5)[0])
        
        # Add vertical time marker
        self.unwrapped_lines_d.append(self.ax_unwrap.axvline(
            x=theta, color='gray', linestyle='--', alpha=0.5))
        
        # Add points for current values
        self.unwrapped_lines_d.append(self.ax_unwrap.plot(
            theta, vd, 'purple', marker='o')[0])
        self.unwrapped_lines_q.append(self.ax_unwrap.plot(
            theta, vq, 'orange', marker='o')[0])
        
        return self.rotating_vectors + self.unwrapped_lines_d + self.unwrapped_lines_q

# Create and run animation
viz = UnwrappingVisualizer()
anim = FuncAnimation(viz.fig, viz.update, frames=60, 
                    interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()