import numpy
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

import drone

nodes = 21
number_of_states = 6
x0 = numpy.zeros((nodes * number_of_states, 1))
initial_conditions = numpy.array([0, 0, 0, 0], dtype=float)
desired_trajectory = numpy.concatenate((numpy.ones((nodes,)), numpy.zeros((nodes,)), numpy.ones((nodes,)), numpy.zeros((nodes,))), axis=0)
iteration_range = 20
history = []

# Initialize Drone Object:
agent = drone.Drone(nodes, initial_conditions, x0, desired_trajectory)

# Initialize Optimization:
agent.initialize_optimization()

for i in range(iteration_range):
    # Update Constraints:
    agent.update_optimization()
    # Generate Trajectory:
    agent.generate_trajectory()
    # Create Solution History:
    history.append(numpy.reshape(agent.solution.x, (nodes, number_of_states), order='F'))

# Setup Animation Writer:
FPS = 20
dpi = 300
writerObj = FFMpegWriter(fps=FPS)

# Create Animation:
# Setup Figure: Initialize Figure / Axe Handles
fig, ax = plt.subplots()
p1, = ax.plot([], [], color='black', linewidth=2)
lb, ub = -5, 5
ax.set_xlim([lb, ub])
ax.set_ylim([lb, ub])
ax.set_xlabel('X')  # X Label
ax.set_ylabel('Y')  # Y Label
ax.set_title('Double Integrator Agent:')
video_title = "simulation_"

# Initialize Patch:
agent_patch = Circle((0, 0), radius=0.1, color='cornflowerblue', zorder=10)
ax.add_patch(agent_patch)

# Goal:
goal_patch = Circle((1, 1), radius=0.1, color='red', zorder=1)
ax.add_patch(goal_patch)

# Plot and Create Animation:
with writerObj.saving(fig, video_title+".mp4", dpi):
    for i in range(iteration_range):
        for j in range(0, nodes):
            # Draw Pendulum Arm:
            agent_patch.center = (history[i][j, 0], history[i][j, 3])
            # Update Drawing:
            fig.canvas.draw()
            # Grab and Save Frame:
            writerObj.grab_frame()