import numpy
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

import drone_risk

import pdb

nodes = 21
number_of_states = 9
x0 = numpy.zeros((nodes * number_of_states, 1))
# initial_conditions Data format: [x, y, z, dx, dy, dz]
initial_conditions = numpy.array([-1, -1, 0, 0, 0, 0], dtype=float)
# desired_trajectory Data format: [x, dx, y, dy, z, dz]
desired_trajectory = numpy.concatenate((numpy.ones((nodes,)), numpy.zeros((nodes,)), numpy.ones((nodes,)),
                                        numpy.zeros((nodes,)), numpy.zeros((nodes,)), numpy.zeros((nodes,))), axis=0)
iteration_range = 50
history = []
trajectory = []
constraint_history = []
y = numpy.zeros((nodes, nodes))
run_time = 0.0
epsilon = 1E-8

# Initialize Drone Object:
agent = drone_risk.Drone_Risk(nodes, initial_conditions, x0, desired_trajectory)

# Initialize Optimization:
agent.initialize_optimization()

for i in range(iteration_range):
    # Update Constraints:
    agent.update_optimization()
    # Generate Trajectory:
    agent.generate_trajectory()
    # Simulate Agent:
    agent.simulate()
    # Create Solution History:
    history.append(agent.simulation_solution.y)
    trajectory.append(agent.position.copy())
    # Run Time:
    run_time = run_time + agent.solution.info.run_time

print(run_time / iteration_range)

"""
Plots:
"""
# Setup Animation Writer:
FPS = 60
dpi = 300
writerObj = FFMpegWriter(fps=FPS)

# Create Animation:
# Setup Figure: Initialize Figure / Axe Handles
fig, ax = plt.subplots()
p1, = ax.plot([], [], color='black', linewidth=2)
p2, = ax.plot([], [], color='red', linewidth=1)
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
goal_patch = Circle((1, 1), radius=0.1, color='green', zorder=1)
ax.add_patch(goal_patch)

# Obstacle:
obstacle_patch = Circle((0, 0), radius=0.1, color='red', zorder=1)
ax.add_patch(obstacle_patch)

# Plot and Create Animation:
with writerObj.saving(fig, video_title + ".mp4", dpi):
    for i in range(iteration_range):
        for j in range(0, 21):
            # Draw Pendulum Arm:
            agent_patch.center = (history[i][0, j], history[i][1, j])
            p1.set_data(trajectory[i][0, :], trajectory[i][1, :])
            # Update Drawing:
            fig.canvas.draw()
            # Grab and Save Frame:
            writerObj.grab_frame()

