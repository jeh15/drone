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
import nDimSplineFit
import os
import imageio.v2
from matplotlib import gridspec

#Returns true if failure, false else
def trueFailFunc(delta):
    SAFE_RAD = 1
    k = 6.9/(SAFE_RAD)
    prob = np.exp(-delta * k)
    if delta > SAFE_RAD:
        prob = 0
    
    if random.random() < prob:
        return True
    return False
    
    
obstSpeed = 0.2
qo_i = np.array([0, 0.7, 0, 0, 0, 0])
qd_i = np.array([0.2, 0.2, 0, 0, 0, 0])
qd_des = np.array([0, 0, 0, 0, 0, 0])


d = 3
numIters = 300
obstLocs = np.zeros((numIters, 2))
droneLocs = np.zeros((numIters, 2))
targLocs = np.zeros((numIters, 2))
sampleGap = 0.1
setup = ObstAvoid.obst_avoid_setup()
prevTraj = []

things = []

numSplines = 3
myModel = []
logModel = []

myModels = []
logModels = []

for i in range(numIters):
    print(i)
    if i <= 5:
        traj = ObstAvoid.next_traj(setup, qd_i, qd_des, qo_i, prevTraj)
    else:
        traj = ObstAvoid.next_traj(setup, qd_i, qd_des, qo_i, prevTraj, logModel)
    
    #print(traj)
    prevTraj = traj
    F = traj[2*d:3*d, :]
    #print(F)
    x = traj[0, :]
    y = traj[1, :]
    thing = np.transpose(np.vstack((x, y)))
    things.append(thing)
    
    
    
    dt = setup['Th']/(setup['Nodes'] - 1)
    numSamps = 10
    qo_i = Simulation.simulate_obst(sampleGap, qo_i, qd_i)
    qd_i = Simulation.simulate_drone(sampleGap, dt, F, traj[d:2*d, 0], traj[0:d, 0], setup['m'], setup['c'], numSamps)
#     print(qo_i)
#     print(qd_i)

    v = qo_i[0:d] - qd_i[0:d]
    delta = math.sqrt(np.dot(v, v))
    failed = trueFailFunc(delta)
    if failed:
        pt = np.array([delta, 0])
    else:
        pt = np.array([delta, 1])
    print(pt)
    
    if i == 0:
        data = pt
    else:
        data = np.vstack((data, pt))
    
    if i >= 5 and i % 5 == 0:
        #print(data)
        myModel = nDimSplineFit.splineFit(data, numSplines, False, True)
        logModel = np.zeros(np.shape(myModel))
        for j in range(np.shape(myModel)[0]):
            logModel[j, 0] = myModel[j, 0]
            logModel[j, 1] = math.log(myModel[j, 1])
        logModel = nDimSplineFit.splineFit(logModel, numSplines, True, False)

    myModels.append(myModel)
    logModels.append(logModel)
#     print(obstLocs[i, :])
    print(qd_i[0], qd_i[1])
    obstLocs[i, 0] = qo_i[0]
    obstLocs[i, 1] = qo_i[1]
#     print(obstLocs[i, :])
    droneLocs[i, 0] = qd_i[0]
    droneLocs[i, 1] = qd_i[1]
    targLocs[i, 0] = qd_des[0]
    targLocs[i, 1] = qd_des[1]

fig = plt.figure()
fig.set_figwidth(4)
fig.set_figheight(8)
spec = gridspec.GridSpec(ncols = 1, nrows = 3, height_ratios = [2, 1, 1])
ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])
ax2 = fig.add_subplot(spec[2])

def animate(i):
    dx, dy = (droneLocs[i, 0], droneLocs[i, 1])
    ox, oy = (obstLocs[i, 0], obstLocs[i, 1])
    tx, ty = (targLocs[i, 0], targLocs[i, 1])
    
    ax0.clear()
    ax0.scatter(dx, dy)
    ax0.scatter(ox, oy)
    #ax0.scatter(tx, ty)
    thing = things[i]
    print(thing)
    ax0.plot(thing[:, 0], thing[:, 1], '--')
    ax0.set_xlim([-2, 2])
    ax0.set_ylim([-2, 2])
    
    myData = data[0:i, :]
    ax1.clear()
    ax1.scatter(myData[:, 0], myData[:, 1])
    mod1 = myModels[i]
    if len(mod1) > 0:
        ax1.plot(mod1[:, 0], mod1[:, 1])
    ax1.set_ylim([0, 1])
    
    ax2.clear()
    mod2 = logModels[i]
    if len(mod2) > 0:
        ax2.plot(mod2[:, 0], mod2[:, 1])
    

# run the animation
ani = FuncAnimation(fig, animate, frames=numIters, interval=10, repeat=False)

plt.show()


#------------ SAVE A GIF -----------------
# filenames = []
# for i in range(numIters):
#     print(i)
#     filename = f'{i}.png'
#     filenames.append(filename)
#     
#     plt.clf()
#     plt.scatter(droneLocs[i, 0], droneLocs[i, 1])
# #     for j in range(len(things[i])):
# #         plt.scatter(things[i][j][0], things[i][j][1], color = 'blue')
#     plt.scatter(obstLocs[i, 0], obstLocs[i, 1])
#     plt.scatter(targLocs[i, 0], targLocs[i, 1])
#     plt.xlim([-1, 1])
#     plt.ylim([-1, 1])
#     plt.savefig(filename)
#     
# # build gif
# with imageio.get_writer('mygif.gif', mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)
#         
# # Remove files
# for filename in set(filenames):
#     os.remove(filename)