import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
myTransform = []

things = []

numSplines = 3
myModel = []
logModel = []

myModels = []
logModels = []
loseStates = []
winStates = []
loseSum = np.zeros(2)
winSum = np.zeros(2)
numWins = 0
numLosses = 0

pastStates = []
pastData = {}

for i in range(numIters):
    traj = ObstAvoid_Two_Sources.next_traj(setup, qd_i, qd_des, qo_i, prevTraj, logModel, myTransform)
    
    prevTraj = traj
    F = traj[2*d:3*d, :]
    x = traj[0, :]
    y = traj[1, :]
    thing = np.transpose(np.vstack((x, y)))
    things.append(thing) #log traj for later plotting
    
    
    
    dt = setup['Th']/(setup['Nodes'] - 1)
    numSamps = 10
    
    #Simulate to obtain new qo and qd
    newQo = Simulation.simulate_obst(sampleGap, qo_i, qd_i)
    newQd = Simulation.simulate_drone(sampleGap, dt, F, traj[d:2*d, 0], traj[0:d, 0], setup['m'], setup['c'], numSamps)

    #use true failure function to determine failure of these states
    failed = trueFailFunc(newQo, newQd)
    
    #Find a "direction vector" from obstacle to drone
    dirVec = (qd_i[0:d] - qo_i[0:d])
    if np.linalg.norm(dirVec) != 0:
        dirVec /= np.linalg.norm(dirVec)
    newVec = newQd - newQo
    
    #Our state currently has two components:
    #1. The projection of the relative position onto the direction vector
    #2. The projection of the drone velocity onto the direction vector
    myState = np.array([np.dot(newVec[:d], dirVec), np.dot(newQd[d:], dirVec)])
    
    #Based on failure, log state as either a loss or success
    if failed:
        print("fail")
        loseStates.append(myState)
        loseSum += myState
        pastStates.append((myState, 0))
    else:
        print("success")
        winStates.append(myState)
        winSum += myState
        pastStates.append((myState, 1))

    #Once we've experienced at least one loss and win, we can start developing the risk model
    if len(loseStates) > 0 and len(winStates) > 0:
        #Find centroids by dividing the running totals
        avgLoss = loseSum/len(loseStates)
        avgWin = winSum/len(winStates)
        
        #Our "transform" is the direction vector from loss centroid to win centroid
        v = avgWin - avgLoss
        myTransform = v
        if np.linalg.norm(myTransform) != 0:
            myTransform /= np.linalg.norm(myTransform)
        
        #Every 5th iteration (to make the MPC run faster), we update the risk model
        if i % 5 == 0:
            #Dot product of transform with state gives overall delta, which is our x-coordinate
            #Our y-coordinate is 1 for succcess and 0 for failure
            data = np.zeros((i + 1, 2))
            for j in range(len(winStates)):
                data[j, 0] = np.dot(myTransform, winStates[j])
                data[j, 1] = 1
            for j in range(len(loseStates)):
                data[len(winStates) + j, 0] = np.dot(myTransform, loseStates[j])
                data[len(winStates) + j, 1] = 0
            
            pastData[i] = data #logging data for plotting
            
            #2-D spline fit on this data, log to obtain log data, then spline fit again with convexity
            myModel = nDimSplineFit.splineFit(data, numSplines, False, True)
            logModel = np.zeros(np.shape(myModel))
            for j in range(np.shape(myModel)[0]):
                logModel[j, 0] = myModel[j, 0]
                logModel[j, 1] = math.log(myModel[j, 1])
            logModel = nDimSplineFit.splineFit(logModel, numSplines, True, False)
            #this is the model we use in the next optimization
    
    #save each model for plotting
    myModels.append(myModel)
    logModels.append(logModel)
    
    #Update qo and qd
    qo_i = newQo
    qd_i = newQd
    
    #Save all the positions for plotting later
    obstLocs[i, 0] = qo_i[0]
    obstLocs[i, 1] = qo_i[1]
    droneLocs[i, 0] = qd_i[0]
    droneLocs[i, 1] = qd_i[1]
    targLocs[i, 0] = qd_des[0]
    targLocs[i, 1] = qd_des[1]


#Animate trajectory
fig = plt.figure()
fig.set_figwidth(8)
fig.set_figheight(8)
spec = gridspec.GridSpec(ncols = 2, nrows = 2, height_ratios = [2, 1])
ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])
ax2 = fig.add_subplot(spec[2])
ax3 = fig.add_subplot(spec[3])

def animate(i):
    print(i)
    
    #Ax0 plots the trajectory itself -- qd and qo
    dx, dy = (droneLocs[i, 0], droneLocs[i, 1])
    ox, oy = (obstLocs[i, 0], obstLocs[i, 1])
    
    ax0.clear()
    ax0.scatter(dx, dy)
    ax0.scatter(ox, oy)
    thing = things[i]
    ax0.plot(thing[:, 0], thing[:, 1], '--')
    ax0.set_xlim([-2, 2])
    ax0.set_ylim([-2, 2])
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    
    #Ax1 plots the successes and failures in the state space, along with an arrow
    #between their centroids
    ax1.clear()
    winSum = np.zeros(2)
    loseSum = np.zeros(2)
    numWin = 0
    numLose = 0
    for j in range(i):
        state = pastStates[j]
        if state[1] == 1:
            ax1.scatter(state[0][0], state[0][1], color = "green")
            winSum += state[0]
            numWin += 1
        else:
            ax1.scatter(state[0][0], state[0][1], color = "red")
            loseSum += state[0]
            numLose += 1
    if numLose > 0 and numWin > 0:
        wAvg = winSum/numWin
        lAvg = loseSum/numLose
        ax1.arrow(lAvg[0], lAvg[1], wAvg[0] - lAvg[0], wAvg[1] - lAvg[1])
    ax1.set_xlabel("Delta 1 (distance)")
    ax1.set_ylabel("Delta 2 (velocity)")
    
    #Ax2 plots the non-log risk model
    ax2.clear()
    if i in pastData:
        myData = pastData[i]
        ax2.scatter(myData[:, 0], myData[:, 1])
    mod1 = myModels[i]
    if len(mod1) > 0:
        ax2.plot(mod1[:, 0], mod1[:, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Transformed delta")
    ax2.set_ylabel("Prob of Success")
    
    #Ax3 plots the logged risk model
    ax3.clear()
    mod2 = logModels[i]
    if len(mod2) > 0:
        ax3.plot(mod2[:, 0], mod2[:, 1])
    ax3.set_xlabel("Transformed delta")
    ax3.set_ylabel("LogS")
    

# run the animation
ani = animation.FuncAnimation(fig, animate, frames=numIters, interval=10, repeat=False)
#ani.save('testAnimation2.gif', writer='imagemagick', fps=20)

plt.show()