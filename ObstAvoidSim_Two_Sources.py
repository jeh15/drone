import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
import time
import ObstAvoid_Two_Sources
import Simulation
import nDimSplineFit
import os
import imageio.v2
from matplotlib import gridspec

#Returns true if failure, false else
def trueFailFunc(qo_i, qd_i):
    v = qo_i[0:d] - qd_i[0:d]
    delta = math.sqrt(np.dot(v, v))
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
qd_i = np.array([0.4, 0.2, 0, 0, 0, 0])
qd_des = np.array([0, 0, 0, 0, 0, 0])




d = 3
numIters = 300
obstLocs = np.zeros((numIters, 2))
droneLocs = np.zeros((numIters, 2))
targLocs = np.zeros((numIters, 2))
sampleGap = 0.1
setup = ObstAvoid_Two_Sources.obst_avoid_setup()
prevTraj = []
transform = []

things = []

numSplines = 3
myModel = []
logModel = []

myModels = []
logModels = []

failedOnce = False
wonOnce = False
loseStates = []
winStates = []
loseSum = np.zeros(2)
winSum = np.zeros(2)

for i in range(numIters):
    print(i)
    traj = ObstAvoid_Two_Sources.next_traj(setup, qd_i, qd_des, qo_i, prevTraj, logModel, transform)
    
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
    newQo = Simulation.simulate_obst(sampleGap, qo_i, qd_i)
    newQd = Simulation.simulate_drone(sampleGap, dt, F, traj[d:2*d, 0], traj[0:d, 0], setup['m'], setup['c'], numSamps)
#     print(qo_i)
#     print(qd_i)
    failed = trueFailFunc(newQo, newQd)
    dirVec = (qd_i[0:d] - qo_i[0:d])
    dirVec /= np.linalg.norm(dirVec)
    newVec = newQd - newQo
    print("newVec", newVec)
    myState = np.array([np.dot(newVec[:d], dirVec), np.dot(newQd[d:], dirVec)])

    if failed:
        print("fail")
        loseStates.append(myState)
        loseSum += myState
        failedOnce = True
    else:
        print("success")
        winStates.append(myState)
        print(winSum)
        print(myState)
        winSum += myState
        wonOnce = True
        
    if failedOnce and wonOnce:
        avgLoss = loseSum/len(loseStates)
        avgWin = winSum/len(winStates)
        v = avgWin - avgLoss
        myTransform = v
        if np.linalg.norm(myTransform) != 0:
            myTransform /= np.linalg.norm(myTransform)
        
        if i % 5 == 0:
            data = np.zeros((i + 1, 2))
            for j in range(len(winStates)):
                data[j, 0] = np.dot(myTransform, winStates[j])
                data[j, 1] = 1
            for j in range(len(loseStates)):
                data[len(winStates) + j, 0] = np.dot(myTransform, loseStates[j])
                data[len(winStates) + j, 1] = 0
            print(data)
            
            myModel = nDimSplineFit.splineFit(data, numSplines, False, True)
            logModel = np.zeros(np.shape(myModel))
            for j in range(np.shape(myModel)[0]):
                logModel[j, 0] = myModel[j, 0]
                logModel[j, 1] = math.log(myModel[j, 1])
            logModel = nDimSplineFit.splineFit(logModel, numSplines, True, False)

    myModels.append(myModel)
    logModels.append(logModel)
    
    qo_i = newQo
    qd_i = newQd
    print(qd_i, qo_i)

    obstLocs[i, 0] = qo_i[0]
    obstLocs[i, 1] = qo_i[1]
    droneLocs[i, 0] = qd_i[0]
    droneLocs[i, 1] = qd_i[1]
    targLocs[i, 0] = qd_des[0]
    targLocs[i, 1] = qd_des[1]

avgWin = sum(winStates)/len(winStates)
avgLose = sum(loseStates)/len(loseStates)
print(avgWin)
print(avgLose)
plt.scatter([w[0] for w in winStates], [w[1] for w in winStates])
plt.scatter([l[0] for l in loseStates], [l[1] for l in loseStates])
plt.xlabel("Delta 1")
plt.ylabel("Delta 2")
plt.show()

fig = plt.figure()
fig.set_figwidth(4)
fig.set_figheight(8)
spec = gridspec.GridSpec(ncols = 1, nrows = 3, height_ratios = [2, 1, 1])
ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])
ax2 = fig.add_subplot(spec[2])

def animate(i):
    print(i)
    dx, dy = (droneLocs[i, 0], droneLocs[i, 1])
    ox, oy = (obstLocs[i, 0], obstLocs[i, 1])
    tx, ty = (targLocs[i, 0], targLocs[i, 1])
    
    ax0.clear()
    ax0.scatter(dx, dy)
    ax0.scatter(ox, oy)
    #ax0.scatter(tx, ty)
    thing = things[i]
    #print(thing)
    ax0.plot(thing[:, 0], thing[:, 1], '--')
    ax0.set_xlim([-2, 2])
    ax0.set_ylim([-2, 2])
    
    myData = data[0:i, :]
    ax1.clear()
    #ax1.scatter(myData[:, 0], myData[:, 1])
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
