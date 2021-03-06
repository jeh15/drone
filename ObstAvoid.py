from ObstAvoidSetup import obst_avoid_setup
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
    setup['Th'] = 2.5;
    setup['Nodes'] = 11;
    setup['r_min'] = 0.1; #hard
    setup['r_min2'] = 0.3; #soft
    setup['cost'] = 5;
    setup['xd_lb'] = np.array([-2, -2, 0]);
    setup['xd_ub'] = np.array([2, 2, 0]);
    setup['vd_lb'] = np.array([-0.25, -0.25, 0]);
    setup['vd_ub'] = np.array([0.25, 0.25, 0]);
    
    m = 0.032;
    c = 0.5;
    n = setup['Nodes']
    Th = setup['Th']
    dt = Th/(n - 1)
    
    H = np.zeros((11*n, 11*n))
    for i in range(3*n):
        H[i, i] = 1
    for i in range(10*n, 11*n):
        H[i, i] = setup['cost']
    H = 2 * H
    setup['H'] = H
    
    ABound = np.zeros((6*n, 11*n))
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
    
    ADyn1 = np.zeros((3*n-3, 11*n))
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
    
    ADyn2 = np.zeros((3*n-3, 11*n))
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
    
    ###Risk Constraints###
    #Right now the "risk model" is just a simple piecewise function that
    #is equal to -delta when delta is negative and equal to 0 when it is
    #positive. But theoretically, this could be replaced with whatever
    #spline model we want relative to delta! Incorporating speed as
    #the second risk source may be a little harder.
    #(Also, if we change the risk model, this can't be precomputed anymore)
    
    ARisk = np.zeros((2*n, 11*n))
    lRisk = np.zeros((2*n, 1))
    uRisk = np.zeros((2*n, 1))
    
    for i in range(n):
        ARisk[i, (11-1)*n + i] = 1
        lRisk[i] = 0
        uRisk[i] = math.inf
    for i in range(n):
        ARisk[n + i, (11 - 1)*n + i] = 1
        ARisk[n + i, (11 - 2)*n + i] = 1
        lRisk[n + i] = 0
        uRisk[n + i] = math.inf
        
    setup['ASetup'] = np.vstack((ABound, ADyn1, ADyn2, ARisk))
    setup['lSetup'] = np.vstack((lBound, lDyn1, lDyn2, lRisk))
    setup['uSetup'] = np.vstack((uBound, uDyn1, uDyn2, uRisk))
    
    return setup

#----------------------------------------------------------------------------------

#This function returns the next trajectory given states of drone, obstacle, and target,
#along with the setup parameters/precomputed matrices
#Optional parameter prevTraj takes into account previous trajectory when computing
#line constraints
#Trajectory is in the form of the 33 x n matrix: dt, x (0-7), y (0-7), z (0-7), yaw (0-7)
def next_traj(setup, qd_i, qd_des, qo_i, prevTraj = []):
    T = setup['Th']
    n = setup['Nodes']
  
    d = qo_i.shape[0]//2
    
    myPos = np.array(qd_i[0:d])
    obstPos = np.array(qo_i[0:d])
    targPos = np.array(qd_des[0:d])
    
    vec = myPos - obstPos
    if np.dot(vec, vec) < (setup['r_min'])**2:
        raise Exception('Minimum distance from obstacle violated, optimization did not produce results')
    
    for i in range(0, d):
        if qd_i[i] < setup['xd_lb'][i] or qd_i[i] > setup['xd_ub'][i]:
            raise Exception('Cartesian Bounds Violated, Optimization did not produce results')
    
    mag = math.sqrt(np.dot(vec, vec))
    unitVec = vec / mag
    
    dt = T/(n - 1)
    lenX = 3 * d + 2
    
    f = np.zeros((lenX*n, 1))
    for i in range(d):
        for j in range(i*n, (i+1)*n):
            f[j] = -2 * targPos[i] 
        
    ###Edge Constraints###
    lEdge = np.zeros((2*d, 1))
    uEdge = np.zeros((2*d, 1))
    AEdge = np.zeros((2*d, lenX*n))
    
    for i in range(2*d):
        AEdge[i, i*n] = 1
        lEdge[i] = qd_i[i]
        uEdge[i] = qd_i[i]
    
    myPosList = np.zeros((3, n))
    if len(prevTraj) != 0:   
        myXList = prevTraj[:, 1].reshape((1, n - 1))
        myYList = prevTraj[:, 9].reshape((1, n - 1))
        myZList = prevTraj[:, 17].reshape((1, n - 1))
        myPosList[0, 0:n-1] = myXList
        myPosList[1, 0:n-1] = myYList
        myPosList[2, 0:n-1] = myZList
        myPosList[:, n-1] = myPosList[:, n-2]
        myPosList[:, 0] = myPos
    
    ###Hard Line Constraints###
    ALine = np.zeros((n, lenX*n))
    lLine = np.zeros((n, 1))
    uLine = np.zeros((n, 1))
    
    op = obstPos;
    obstVel = np.array(qo_i[d : 2*d])
    obstSpeed = math.sqrt(np.dot(obstVel, obstVel))
    for i in range(n):
        if len(prevTraj) != 0:
            myPos = myPosList[:, i]
        #print(myPos)
        op = op + obstVel * dt
        vec = myPos - op
        mag = math.sqrt(np.dot(vec, vec))
        unitVec = vec/mag
        
        pt = op + unitVec * setup['r_min']
        for j in range(d):
            ALine[i, j*n + i] = unitVec[j]
        lLine[i] = np.dot(unitVec, pt)
        uLine[i] = math.inf
        
    
    ###Delta Definition Constraints###
    ADelta = np.zeros((n, lenX*n))
    lDelta = np.zeros((n, 1))
    uDelta = np.zeros((n, 1))
    op = obstPos;
    
    factor = 0
    for i in range(n):
        if len(prevTraj) != 0:
            myPos = myPosList[:, i]
        op = op + obstVel * dt
        vec = myPos - op
        mag = math.sqrt(np.dot(vec, vec))
        unitVec = vec/mag
        
        pt = op + unitVec * setup['r_min2']
        for j in range(d):
            ADelta[i, j*n + i] = unitVec[j]
        ADelta[i, (lenX-2)*n + i] = -1
        lDelta[i] = np.dot(unitVec, pt)
        uDelta[i] = np.dot(unitVec, pt)
    
    ###Solve Optimization###
    ASetup = setup['ASetup']
    a1 = np.shape(ASetup)[0]
    a2 = np.shape(AEdge)[0]
    a3 = np.shape(ALine)[0]
    a4 = np.shape(ADelta)[0]
    height = a1 + a2 + a3 + a4
    A = np.zeros((height, lenX*n))
    l = np.zeros((height, 1))
    u = np.zeros((height, 1))
    
    A[0:a1, :] = setup['ASetup']
    A[a1:a1+a2, :] = AEdge
    A[a1+a2:a1+a2+a3, :] = ALine
    A[a1+a2+a3:, :] = ADelta
    
    l[0:a1, :] = setup['lSetup']
    l[a1:a1+a2, :] = lEdge
    l[a1+a2:a1+a2+a3, :] = lLine
    l[a1+a2+a3:, :] = lDelta
    
    u[0:a1, :] = setup['uSetup']
    u[a1:a1+a2, :] = uEdge
    u[a1+a2:a1+a2+a3, :] = uLine
    u[a1+a2+a3:, :] = uDelta
    
    H = sparse.csc_matrix(setup['H'])
    A = sparse.csc_matrix(A)
    st = time.time()
    prob = osqp.OSQP()
    
    
    prob.setup(H, f, A, l, u, warm_start = True)
    res = prob.solve()
    #print(time.time() - st)
    sol = res.x
    #print(sol)
    traj = np.zeros((lenX, n))
    for i in range(lenX):
        traj[i, :] = np.transpose(sol[i*n : (i+1)*n])
    #print(traj)
    
    
    finalTraj = np.zeros((n, 33))
    finalTraj[:, 0] = dt
    for i in range(3):
        finalTraj[:, 8*i + 1] = traj[i, :]
        finalTraj[:, 8*i + 2] = traj[i + 3, :]
        for j in range(3, 9):
            finalTraj[:, 8*i + j] = np.gradient(finalTraj[:, 8*i + j - 1])
    finalTraj = finalTraj[1:, :]
    
    return finalTraj

#-------------------------------------------------------------------------------------
### TESTING CODE ###
# setup = obst_avoid_setup()
# 
# qd_i = np.array([0.2, 0.2, 0, 0, 0, 0]);
# qd_des = np.array([0, 0, 0, 0, 0, 0]);
# qo_i = np.array([0, 1, 0, 0, -0.4, 0]);
# 
# fig, ax = plt.subplots(2)
# 
# startTime = time.time()
# traj = next_traj(setup, qd_i, qd_des, qo_i) #Without previous trajectory
# print(time.time() - startTime)
# ax[0].plot(traj[:, 1], traj[:, 9])
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# 
# 
# startTime = time.time()
# traj = next_traj(setup, qd_i, qd_des, qo_i, traj) #With previous trajectory
# print(time.time() - startTime)
# ax[1].plot(traj[:, 1], traj[:, 9])
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.show()  
    
