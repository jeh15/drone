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

# Initialize Risk Regression:
x_data = numpy.linspace(-1, 1, 21)
# x_data[-2] = 1
# y_data = numpy.random.rand(21)
y_data = numpy.array(
    [0.32047225, 0.64431375, 0.82075674, 0.12774672, 0.92728747, 0.64255318, 0.65090247, 0.35466653, 0.21408264,
     0.21432659, 0.51158198, 0.870237, 0.77179843, 0.09303753, 0.32216978, 0.40071889, 0.63870181, 0.89355759,
     0.30076709, 0.68252839, 0.98761149])
agent.risk_sample = numpy.vstack((x_data, y_data))
agent.initialize_fpf()
print('FPF')
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
