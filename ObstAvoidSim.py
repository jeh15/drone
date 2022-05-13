import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
import time
import ObstAvoid
import Simulation

obstSpeed = 0.2
qo_i = np.array([0, 1, 0, 0, 0, 0])
qd_i = np.array([0.2, 0.2, 0, 0, 0, 0])
qd_des = np.array([0, 0, 0, 0, 0, 0])


d = 3
numIters = 120
obstLocs = np.zeros((numIters, 2))
droneLocs = np.zeros((numIters, 2))
targLocs = np.zeros((numIters, 2))
sampleGap = 0.1
setup = ObstAvoid.obst_avoid_setup()
prevTraj = []

things = []


for i in range(numIters):
    print(i)
    traj = ObstAvoid.next_traj(setup, qd_i, qd_des, qo_i, prevTraj)
    #print(traj)
    prevTraj = traj
    F = traj[2*d:3*d, :]
    print(F)
    x = traj[0, :]
    y = traj[1, :]
    thing = [[x[i], y[i]] for i in range(len(x))]
    things.append(thing)
    
    
    dt = setup['Th']/(setup['Nodes'] - 1)
    numSamps = 10
    qo_i = Simulation.simulate_obst(sampleGap, qo_i, qd_i)
    qd_i = Simulation.simulate_drone(sampleGap, dt, F, traj[d:2*d, 0], traj[0:d, 0], setup['m'], setup['c'], numSamps)
#     print(qo_i)
#     print(qd_i)
#     print(obstLocs[i, :])
    obstLocs[i, 0] = qo_i[0]
    obstLocs[i, 1] = qo_i[1]
#     print(obstLocs[i, :])
    droneLocs[i, 0] = qd_i[0]
    droneLocs[i, 1] = qd_i[1]
    targLocs[i, 0] = qd_des[0]
    targLocs[i, 1] = qd_des[1]

plt.ion()
for i in range(numIters):
    plt.clf()
    plt.scatter(droneLocs[i, 0], droneLocs[i, 1])
#     for j in range(len(things[i])):
#         plt.scatter(things[i][j][0], things[i][j][1], color = 'blue')
    plt.scatter(obstLocs[i, 0], obstLocs[i, 1])
    plt.scatter(targLocs[i, 0], targLocs[i, 1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
    plt.pause(0.00001)
    

# allLocs = np.hstack((obstLocs, droneLocs, targLocs))
# # print(allLocs)
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')
# 
# def init():
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     return ln,
# 
# def update(frame):
#     print(frame)
#     xdata = [frame[0], frame[2], frame[4]]
#     ydata = [frame[1], frame[3], frame[5]]
#     ln.set_data(xdata, ydata)
#     return ln,
# 
# 
# ani = FuncAnimation(fig, update, allLocs,
#                     init_func=init, blit=True, interval = 50, repeat = False)
# plt.show()

# fig = plt.figure(figsize=(7,5))
# 
# def animation_function(i):
#     plt.scatter(droneLocs[i, 0], droneLocs[i, 1], color = 'blue')
#     plt.scatter(obstLocs[i, 0], obstLocs[i, 1], color = 'red')
#     plt.scatter(targLocs[i, 0], targLocs[i, 1], color = 'green')
#     x = [things[i][j][0] for j in range(len(things[i]))]
#     y = [things[i][j][1] for j in range(len(things[i]))]
#     plt.scatter(x, y, color = 'blue', alpha = 0.5)
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     
#     
# animation = FuncAnimation(fig, animation_function, 
#                           interval = 50, frames = (0, numIters))
# 
# plt.show()
    