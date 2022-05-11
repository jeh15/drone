import numpy
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
# import drone_drake_test
import drone_matlab
import pdb

nodes = 21
# nodes = 11
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
risk_history = []
fpf_history = []
ls_history = []

# Setup Animation Writer:
FPS = 60
dpi = 300
writerObj = FFMpegWriter(fps=FPS)
video_title = "simulation_risk_learning_"

# Setup Figure: Initialize Figure / Axe Handles
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 4)

trajectory_plot, = ax1.plot([], [], color='black', linewidth=2)
constraint_plot, = ax1.plot([], [], color='red', linewidth=2)
p1, = ax2.plot([], [], marker=".", color='black', linewidth=0)
p2, = ax2.plot([], [], color='red', linewidth=1)
p3, = ax3.plot([], [], marker=".", color='black', linewidth=0)
p4, = ax3.plot([], [], color='red', linewidth=1)

ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax2.set_xlim([-2, 2])
ax2.set_ylim([0, 1.5])
ax2.set_ylabel('Failure Probability')
ax2.set_xlabel('Delta')
ax3.set_xlim([-2, 2])
ax3.set_ylabel('Log-Survival')
ax3.set_xlabel('Delta')

# Initialize Patch:
agent_patch = Circle((0, 0), radius=0.1, color='cornflowerblue', zorder=10)
ax1.add_patch(agent_patch)

# Goal:
goal_patch = Circle((1, 1), radius=0.1, color='green', zorder=1)
ax1.add_patch(goal_patch)

# Obstacle:
obstacle_patch = Circle((0, 0), radius=0.1, color='red', zorder=1)
obstacle_risk_patch = Circle((0, 0), radius=0.5, edgecolor='red', facecolor='none', linestyle='--', zorder=1)
ax1.add_patch(obstacle_patch)
ax1.add_patch(obstacle_risk_patch)

# Initialize Drone Object:
agent = drone_matlab.Drone_Risk(nodes, initial_conditions, x0, desired_trajectory)

# Initialize Optimization:
# agent.initialize_optimization()
# agent.update_optimization()
# agent.generate_trajectory()

# # Initialize Risk Regression:
x_data = numpy.linspace(-1, 1, 21)
# x_data[-2] = 1
# y_data = numpy.random.rand(21)
y_data = numpy.array(
    [0.32047225, 0.64431375, 0.82075674, 0.12774672, 0.92728747, 0.64255318, 0.65090247, 0.35466653, 0.21408264,
     0.21432659, 0.51158198, 0.870237, 0.77179843, 0.09303753, 0.32216978, 0.40071889, 0.63870181, 0.89355759,
     0.30076709, 0.68252839, 0.98761149])
agent.risk_sample = numpy.vstack((x_data, y_data))
print('FPF')
agent.initialize_fpf()
agent.get_fpf()
print('LS')
agent.initialize_ls()
agent.get_ls()


# Setup Figure: Initialize Figure / Axe Handles
fig, ax = plt.subplots(2, 1)
p1, = ax[0].plot([], [], marker=".", color='black', linewidth=0)
p2, = ax[0].plot([], [], color='red', linewidth=1)
p3, = ax[1].plot([], [], marker=".", color='black', linewidth=0)
p4, = ax[1].plot([], [], color='red', linewidth=1)
lb, ub = -1, 1
ax[0].set_xlim([lb, ub])
ax[0].set_ylim([lb, ub])
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[1].set_xlim([lb, ub])
ax[1].set_ylim([lb, ub])
ax[1].set_xlabel('X')
ax[1].set_ylabel('Log-Y')

p1.set_data(x_data, y_data)
p2.set_data(agent.fpf_x, agent.fpf_y)

p3.set_data(agent.ls_x, agent.ls_y)
p4.set_data(agent.ls_x, agent.ls_y)

plt.show()

