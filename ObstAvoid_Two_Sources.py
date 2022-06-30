import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import math
import random
import time

#-------------------------------------------------------------------------------------
#Running this function returns a dictionary "setup" that contains all
#the setup values necessary for any future optimizations.
#This includes parameters like Th, Nodes, r_min, r_min2, etc.
#Also includes all matrices that can be precomputed without knowledge of
#states -- H, ABound, ADyn1, ADyn2, and ARisk for now
def obst_avoid_setup():
    setup = {}
    setup['Th'] = 2.5
    setup['Nodes'] = 11
    setup['r_min'] = 0 #hard
    setup['r_min2'] = 0 #soft
    setup['cost'] = 100
    setup['xd_lb'] = np.array([-2, -2, 0])
    setup['xd_ub'] = np.array([2, 2, 0])
    setup['vd_lb'] = np.array([-0.25, -0.25, 0])
    setup['vd_ub'] = np.array([0.25, 0.25, 0])
    
    m = 0.032
    c = 0.5
    setup['m'] = m
    setup['c'] = c
    
    n = setup['Nodes']
    Th = setup['Th']
    dt = Th/(n - 1)
    
    H = np.zeros((12*n, 12*n))
    #For distance from target
#     for i in range(3*n):
#         H[i, i] = 1
    H = 2 * H
    setup['H'] = H
    
    ABound = np.zeros((6*n, 12*n))
    lBound = np.zeros((6*n, 1))
    uBound = np.zeros((6*n, 1))
    
    #xd_lb < x < xd_ub
    for i in range(3):
        for j in range(n):
            ABound[i*n + j, i*n + j] = 1
            lBound[i*n + j] = setup['xd_lb'][i]
            uBound[i*n + j] = setup['xd_ub'][i]
    
    #vd_lb < v < vd_ub
    for i in range(3):
        for j in range(n):
            ABound[(i+3)*n + j, (i+3)*n + j] = 1
            lBound[(i+3)*n + j] = setup['vd_lb'][i]
            uBound[(i+3)*n+j] = setup['vd_ub'][i]
    
    ###First-order dynamic constraints###
    
    ADyn1 = np.zeros((3*n-3, 12*n))
    lDyn1 = np.zeros((3*n-3, 1))
    uDyn1 = np.zeros((3*n-3, 1))
    
    #x1 = x0 + dx0 * T/n --> x1 - x0 - T/n dx0 = 0
    for i in range(3):
        for j in range(n - 1):
            ADyn1[i*(n-1) + j, i*n + j] = -1
            ADyn1[i*(n-1) + j, i*n + j + 1] = 1
            ADyn1[i*(n-1) + j, (i+3)*n + j] = -dt
            lDyn1[i*(n-1) + j] = 0
            uDyn1[i*(n-1) + j] = 0
    
    ###Second-order dynamic constraints###
    
    ADyn2 = np.zeros((3*n-3, 12*n))
    lDyn2 = np.zeros((3*n-3, 1))
    uDyn2 = np.zeros((3*n-3, 1))
    
    #dx1 = dx0 + T/n * (u0/m - c*dx0)
    #dx1 = (1 - cT/n) dx0 + (T/mn) u0
    #dx1 + (cT/n - 1)dx0 - (T/mn) u0 = 0
    for i in range(3):
        for j in range(n - 1):
            ADyn2[i*(n-1) + j, (i+3)*n + j] = c*dt - 1
            ADyn2[i*(n-1) + j, (i+3)*n + j + 1] = 1
            ADyn2[i*(n-1) + j, (i+2*3)*n + j] = -dt/m
            lDyn2[i*(n-1) + j] = 0
            uDyn2[i*(n-1) + j] = 0
    
        
    setup['ASetup'] = np.vstack((ABound, ADyn1, ADyn2))
    setup['lSetup'] = np.vstack((lBound, lDyn1, lDyn2))
    setup['uSetup'] = np.vstack((uBound, uDyn1, uDyn2))
    
    return setup

#----------------------------------------------------------------------------------

#This function returns the next trajectory given states of drone, obstacle, and target,
#along with the setup parameters/precomputed matrices
#Optional parameter prevTraj takes into account previous trajectory when computing
#line constraints
#Trajectory is in the form of the 33 x n matrix: dt, x (0-7), y (0-7), z (0-7), yaw (0-7)
def next_traj(setup, qd_i, qd_des, qo_i, prevTraj = [], model = [], transform = []):
    T = setup['Th']
    n = setup['Nodes']
  
    d = qo_i.shape[0]//2
    
    myPos = np.array(qd_i[0:d])
    obstPos = np.array(qo_i[0:d])
    targPos = np.array(qd_des[0:d])
    
    vec = myPos - obstPos
#     if np.dot(vec, vec) < (setup['r_min'])**2:
#         raise Exception('Minimum distance from obstacle violated, optimization did not produce results')
#     
    for i in range(0, d):
        if qd_i[i] < setup['xd_lb'][i] or qd_i[i] > setup['xd_ub'][i]:
            raise Exception('Cartesian Bounds Violated, Optimization did not produce results')
    
    mag = math.sqrt(np.dot(vec, vec))
    unitVec = vec / mag
    
    dt = T/(n - 1)
    lenX = 3 * d + 3
    
    f = np.zeros((lenX*n, 1))
    #For distance from target
#     for i in range(d):
#         for j in range(i*n, (i+1)*n):
#             f[j] = -2 * targPos[i]
    for i in range((lenX-1)*n, lenX*n):
        f[i] = -1 #The current objective is simply to maximize sum of logS (minimize risk)
        
        
    ###Edge Constraints###
    lEdge = np.zeros((2*d, 1))
    uEdge = np.zeros((2*d, 1))
    AEdge = np.zeros((2*d, lenX*n))
    
    for i in range(2*d):
        AEdge[i, i*n] = 1
        lEdge[i] = qd_i[i]
        uEdge[i] = qd_i[i]
    
    #We map a position list to the previous trajectory
    myPosList = np.zeros((3, n))
    if len(prevTraj) != 0:   
        myXList = prevTraj[0, 1:]
        myYList = prevTraj[1, 1:]
        myZList = prevTraj[2, 1:]
        myPosList[0, 0:n-1] = myXList
        myPosList[1, 0:n-1] = myYList
        myPosList[2, 0:n-1] = myZList
        myPosList[:, n-1] = myPosList[:, n-2]
        myPosList[:, 0] = myPos
        
    
    ###Delta Definition Constraints###
    #DeltaPos is (pos - obstPos) projected onto (pos0 - obstPos)
    #DeltaVel is (vel) projected onto (pos0 - obstPos)?
    ADeltaPos = np.zeros((n, lenX*n))
    ADeltaVel = np.zeros((n, lenX*n))
    lDeltaPos = np.zeros((n, 1))
    lDeltaVel = np.zeros((n, 1))
    uDeltaPos = np.zeros((n, 1))
    uDeltaVel = np.zeros((n, 1))
    op = obstPos
    ov = obstVel
    
    for i in range(n):
        if len(prevTraj) != 0:
            myPos = myPosList[:, i] #Replace initial position with pos from prev traj if applicable
        op = op + ov * dt
        unitVec = (myPos - op)/(np.linalg.norm(myPos - op))
        
        pt = op + unitVec * setup['r_min2']
        for j in range(d):
            ADeltaPos[i, j*n + i] = unitVec[j]
            ADeltaVel[i, (j + d)*n + i] = unitVec[j]
        ADeltaPos[i, (lenX-3)*n + i] = -1
        ADeltaVel[i, (lenX-2)*n + i] = -1
        lDeltaPos[i] = np.dot(unitVec, pt)
        lDeltaVel[i] = 0
        uDeltaPos[i] = lDeltaPos[i]
        uDeltaVel[i] = lDeltaVel[i]
    
    ###delta_final = transform[0] * deltaPos + transform[1] * deltaVel
    # r < delta * m_i + b_i for all (m_i, b_i) splines
    ###Risk Constraints###
    numPoints = np.shape(model)[0]
    numSplines = max(numPoints - 1, 0)
    ARisk = np.zeros((numSplines*n, lenX*n))
    lRisk = np.zeros((numSplines*n, 1))
    uRisk = np.zeros((numSplines*n, 1))
    
    if len(model) != 0 and len(transform) != 0:
        for i in range(numSplines):
            m = (model[i+1, 1] - model[i, 1])/(model[i+1,0]-model[i,0])
            b = model[i,1] - m * model[i,0]
            for j in range(n):
                #r < m * delta + b --> m * delta - r > -b
                ARisk[i*n + j, (lenX-3)*n + j] = m * transform[0]
                ARisk[i*n + j, (lenX-2)*n + j] = m * transform[1]
                ARisk[i*n + j, (lenX-1)*n + j] = -1
                lRisk[i*n + j, 0] = -b
                uRisk[i*n + j, 0] = math.inf
    
    #r < 0
    Aneg = np.zeros((n, lenX*n))
    lneg = np.zeros((n, 1))
    uneg = np.zeros((n, 1))
    for j in range(n):
        Aneg[j, (lenX-1)*n + j] = 1
        lneg[j, 0] = -math.inf
        uneg[j, 0] = 0
    
    
    ###Solve Optimization###
    ASetup = setup['ASetup']
    
    A = np.vstack((ASetup, AEdge, ADeltaPos, ADeltaVel, ARisk, Aneg))
    l = np.vstack((setup['lSetup'], lEdge, lDeltaPos, lDeltaVel, lRisk, lneg))
    u = np.vstack((setup['uSetup'], uEdge, uDeltaPos, uDeltaVel, uRisk, uneg))
    
    H = sparse.csc_matrix(setup['H'])
    A = sparse.csc_matrix(A)
    st = time.time()
    prob = osqp.OSQP()
    
    
    prob.setup(H, f, A, l, u, warm_start = True, verbose = True, max_iter = 10000)
    res = prob.solve()
    #print(time.time() - st)
    sol = res.x
    #print(sol)
    traj = np.zeros((lenX, n))
    for i in range(lenX):
        traj[i, :] = np.transpose(sol[i*n : (i+1)*n])
    print(traj)
    return traj
    
    #This code is for transforming into the n x 33 matrix that is used in drone control
#     finalTraj = np.zeros((n, 33))
#     finalTraj[:, 0] = dt
#     for i in range(3):
#         finalTraj[:, 8*i + 1] = traj[i, :]
#         finalTraj[:, 8*i + 2] = traj[i + 3, :]
#         for j in range(3, 9):
#             finalTraj[:, 8*i + j] = np.gradient(finalTraj[:, 8*i + j - 1])
#     finalTraj = finalTraj[1:, :]
#     
#     return finalTraj

#-------------------------------------------------------------------------------------
# ### TESTING CODE ###
# setup = obst_avoid_setup()
# # 
# qd_i = np.array([0.2, 0.2, 0, 0, 0, 0]);
# qd_des = np.array([0, 0, 0, 0, 0, 0]);
# qo_i = np.array([0, 1, 0, 0, -0.3, 0]);
# 
# fig, ax = plt.subplots(1)
# 
# startTime = time.time()
# traj = next_traj(setup, qd_i, qd_des, qo_i) #Without previous trajectory
# print(time.time() - startTime)
# plt.plot(traj[0, :], traj[1, :])
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.show()
# 
# startTime = time.time()
# traj = next_traj(setup, qd_i, qd_des, qo_i, traj) #With previous trajectory
# print(time.time() - startTime)
# ax[1].plot(traj[:, 1], traj[:, 9])
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.show()  
    
