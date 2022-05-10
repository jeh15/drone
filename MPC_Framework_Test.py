import numpy
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import drone_risk
import pdb


# Risk Simulation Helper Function:
def failure_simulation(agent, adversary):
    # Constants
    _r = 0.5
    _scale = 1.0
    # Agent and Adversary Position:
    _agent_x, _agent_y = agent
    _adversary_x, adversary_y = adversary
    # Distance from Adversary
    _d = numpy.sqrt(((_agent_x - _adversary_x) ** 2 + (_agent_y - adversary_y) ** 2)) - _r
    # Failure Probability Simulation
    _rfun = 1 - numpy.exp(_scale * _d)
    _rfun = numpy.where(_rfun < 0, 0, _rfun)
    _simulated_failure = numpy.zeros(_rfun.shape, dtype=float)
    for _i in range(len(_rfun)):
        _simulated_failure[_i] = numpy.random.choice([0, 1], 1, p=[1 - _rfun[_i], _rfun[_i]])
    # First Index of Failure:
    _array = numpy.array([numpy.where(_simulated_failure == 1)])
    if _array.size == 0:
        _idx = len(_d)
        _rfun_x = _d[:]
        _rfun_y = _simulated_failure[:]
        _failure_flag = 0
    else:
        _idx = _array.min()
        _rfun_x = _d[:_idx + 1]
        _rfun_y = _simulated_failure[:_idx + 1]
        _failure_flag = 1
    return _rfun_x, _rfun_y, _idx, _failure_flag


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
agent = drone_risk.Drone_Risk(nodes, initial_conditions, x0, desired_trajectory)

# Initialize Optimization:
agent.initialize_optimization()
# Naive Initialization:
agent.risk_sample = numpy.array([numpy.linspace(0, 1, nodes), numpy.random.rand(nodes)], dtype=float)
agent.initialize_fpf()
agent.get_fpf()
agent.initialize_ls()
agent.get_ls()

# What if we only learn when we detect a failure?
failure_count = 0
failure_flag = 0

# MPC Loop:
for i in range(iteration_range):
    ## DEBUG:
    # if failure_flag == 1:
    #     pdb.set_trace()

    # Update Constraints:
    agent.update_optimization()

    ## DEBUG PLOT:
    # Plot Plane Constraints: (Line)
    x = numpy.linspace(-5, 5, nodes)
    nx, ny, nz = (numpy.diag(agent.plot_A[:, :nodes]), numpy.diag(agent.plot_A[:, 3 * nodes:4 * nodes]),
                  numpy.diag(agent.plot_A[:, 6 * nodes:7 * nodes]))
    ny.setflags(write=1)
    ny[ny == 0] = epsilon
    for j in range(len(nx)):
        y[j, :] = (agent.plot_b[j] - nx[j] * x) / ny[j]
    constraint_history.append(y.copy())
    ##

    # Generate Trajectory:
    print('Trajectory Optimization:')
    agent.generate_trajectory()
    if agent.solution.info.status_val != 1 and agent.solution.info.status_val != 2:
        break_iter = i
        break
    # Simulate Agent:
    agent.simulate()
    # Risk Learning:
    [risk_x, risk_y, idx, failure_flag] = failure_simulation(
        (agent.simulation_solution.y[0, :], agent.simulation_solution.y[1, :]), (0.0, 0.0))
    agent.failure_flag = failure_flag
    if i == 0:
        agent.risk_sample = numpy.vstack((risk_x, risk_y))
    else:
        agent.risk_sample = numpy.hstack((agent.risk_sample, numpy.vstack((risk_x, risk_y))))
    failure_count = failure_count + failure_flag
    agent.failure_counter = failure_count
    print('Failure Probability:')
    agent.get_fpf()
    print('Log-Survival:')
    agent.get_ls()
    if failure_count > 0:
        agent.get_risk_func()
    # Create Solution History:
    history.append(agent.simulation_solution.y[:, :idx])
    trajectory.append(agent.position.copy())
    risk_history.append(agent.risk_sample)
    fpf_history.append([agent.fpf_x.copy(), agent.fpf_y.copy()])
    ls_history.append([agent.ls_x.copy(), agent.ls_y.copy()])
    # Failure Flag Check:
    if failure_flag == 1:
        initial_conditions = numpy.array([-1, -1, 0, 0, 0, 0], dtype=float)
        agent.position = numpy.einsum("ij, i->ij", numpy.ones((3, nodes)), initial_conditions[:3])
        agent.initial_condition[:] = initial_conditions

# Plot and Create Animation:
with writerObj.saving(fig, video_title + ".mp4", dpi):
    for i in range(0, break_iter-1):
        # Update Regression Plots:
        p1.set_data(risk_history[i][0, :], risk_history[i][1, :])
        p2.set_data(fpf_history[i][0], fpf_history[i][1])
        p3.set_data(ls_history[i][0], ls_history[i][1])
        p4.set_data(ls_history[i][0], ls_history[i][1])
        iter_range = len(history[i][0, :])
        for j in range(0, iter_range):
            # Draw Pendulum Arm:
            agent_patch.center = (history[i][0, j], history[i][1, j])
            trajectory_plot.set_data(trajectory[i][0, :], trajectory[i][1, :])
            constraint_plot.set_data(x, constraint_history[i][j, :])
            # Update Drawing:
            fig.canvas.draw()
            # Grab and Save Frame:
            writerObj.grab_frame()
