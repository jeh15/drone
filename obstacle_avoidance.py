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
#This includes parameters like Th, nodes, r_min, r_min2, etc.
#Also includes all matrices that can be precomputed without knowledge of
#states -- H, A_bound, ADyn1, ADyn2, and ARisk for now
def obstacle_avoidance_setup():
    setup = {}
    setup['time_horizon'] = 2.5
    setup['nodes'] = 11
    setup['radius_hard'] = 0.1
    setup['radius_soft'] = 0.3 
    setup['cost'] = 5
    setup['xd_lb'] = np.array([-2, -2, 0], dtype=float)
    setup['xd_ub'] = np.array([2, 2, 0], dtype=float)
    setup['vd_lb'] = np.array([-0.25, -0.25, 0], dtype=float)
    setup['vd_ub'] = np.array([0.25, 0.25, 0], dtype=float)

    # Are these Drone mass and damping parameters?
    m = 0.032
    c = 0.5

    nodes = setup['nodes']
    time_horizon = setup['time_horizon']
    dt = time_horizon / (nodes - 1)
    
    H = np.zeros((11*nodes, 11*nodes))
    for i in range(3*nodes):
        H[i, i] = 1
    for i in range(10*nodes, 11*nodes):
        H[i, i] = setup['cost']
    H = 2 * H
    setup['H'] = H
    
    A_bound = np.zeros((6*nodes, 11*nodes))
    lower_bound = np.zeros((6*nodes, 1))
    upper_bound = np.zeros((6*nodes, 1))
    
    #xd_lb < x < xd_ub
    for i in range(3):
        for j in range(nodes):
            A_bound[i*nodes + j, i*nodes + j] = 1
            lower_bound[i*nodes + j] = setup['xd_lb'][i]
            upper_bound[i*nodes + j] = setup['xd_ub'][i]
    
    #vd_lb < v < vd_ub
    for i in range(3):
        for j in range(nodes):
            A_bound[(i+3)*nodes + j, (i+3)*nodes + j] = 1
            lower_bound[(i+3)*nodes + j] = setup['vd_lb'][i]
            upper_bound[(i+3)*nodes+j] = setup['vd_ub'][i]
    
    ###First-order dynamic constraints###
    
    ADyn1 = np.zeros((3*nodes-3, 11*nodes))
    lDyn1 = np.zeros((3*nodes-3, 1))
    uDyn1 = np.zeros((3*nodes-3, 1))
    
    #x1 = x0 + dx0 * T/nodes --> x1 - x0 - T/nodes dx0 = 0
    for i in range(3):
        for j in range(nodes - 1):
            ADyn1[i*(nodes-1) + j, i*nodes + j] = -1
            ADyn1[i*(nodes-1) + j, i*nodes + j + 1] = 1
            ADyn1[i*(nodes-1) + j, (i+3)*nodes + j] = -dt
            lDyn1[i*(nodes-1) + j] = 0
            uDyn1[i*(nodes-1) + j] = 0
    
    ###Second-order dynamic constraints###
    
    ADyn2 = np.zeros((3*nodes-3, 11*nodes))
    lDyn2 = np.zeros((3*nodes-3, 1))
    uDyn2 = np.zeros((3*nodes-3, 1))
    
    #dx1 = dx0 + T/nodes * (u0/m - c*dx0)
    #dx1 = (1 - cT/nodes) dx0 + (T/mn) u0
    #dx1 + (cT/nodes - 1)dx0 - (T/mn) u0 = 0
    for i in range(3):
        for j in range(nodes - 1):
            ADyn2[i*(nodes-1) + j, (i+3)*nodes + j] = c*dt - 1
            ADyn2[i*(nodes-1) + j, (i+3)*nodes + j + 1] = 1
            ADyn2[i*(nodes-1) + j, (i+2*3)*nodes + j] = -dt/m
            lDyn2[i*(nodes-1) + j] = 0
            uDyn2[i*(nodes-1) + j] = 0
    
    ###Risk Constraints###
    #Right now the "risk model" is just a simple piecewise function that
    #is equal to -delta when delta is negative and equal to 0 when it is
    #positive. But theoretically, this could be replaced with whatever
    #spline model we want relative to delta! Incorporating speed as
    #the second risk source may be a little harder.
    #(Also, if we change the risk model, this can't be precomputed anymore)
    
    ARisk = np.zeros((2*nodes, 11*nodes))
    lRisk = np.zeros((2*nodes, 1))
    uRisk = np.zeros((2*nodes, 1))
    
    for i in range(nodes):
        ARisk[i, (11-1)*nodes + i] = 1
        lRisk[i] = 0
        uRisk[i] = math.inf
    for i in range(nodes):
        ARisk[nodes + i, (11 - 1)*nodes + i] = 1
        ARisk[nodes + i, (11 - 2)*nodes + i] = 1
        lRisk[nodes + i] = 0
        uRisk[nodes + i] = math.inf
        
    setup['ASetup'] = np.vstack((A_bound, ADyn1, ADyn2, ARisk))
    setup['lSetup'] = np.vstack((lower_bound, lDyn1, lDyn2, lRisk))
    setup['uSetup'] = np.vstack((upper_bound, uDyn1, uDyn2, uRisk))
    
    return setup

#----------------------------------------------------------------------------------

#This function returns the next trajectory given states of drone, obstacle, and target,
#along with the setup parameters/precomputed matrices
#Optional parameter prevTraj takes into account previous trajectory when computing
#line constraints
#Trajectory is in the form of the 33 x nodes matrix: dt, x (0-7), y (0-7), z (0-7), yaw (0-7)
def next_trajectory(setup, qd_i, qd_des, qo_i, prevTraj = []):
    T = setup['Th']
    nodes = setup['nodes']
  
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
    
    dt = T/(nodes - 1)
    lenX = 3 * d + 2
    
    f = np.zeros((lenX*nodes, 1))
    for i in range(d):
        for j in range(i*nodes, (i+1)*nodes):
            f[j] = -2 * targPos[i] 
        
    ###Edge Constraints###
    lEdge = np.zeros((2*d, 1))
    uEdge = np.zeros((2*d, 1))
    AEdge = np.zeros((2*d, lenX*nodes))
    
    for i in range(2*d):
        AEdge[i, i*nodes] = 1
        lEdge[i] = qd_i[i]
        uEdge[i] = qd_i[i]
    
    myPosList = np.zeros((3, nodes))
    if len(prevTraj) != 0:   
        myXList = prevTraj[:, 1].reshape((1, nodes - 1))
        myYList = prevTraj[:, 9].reshape((1, nodes - 1))
        myZList = prevTraj[:, 17].reshape((1, nodes - 1))
        myPosList[0, 0:nodes-1] = myXList
        myPosList[1, 0:nodes-1] = myYList
        myPosList[2, 0:nodes-1] = myZList
        myPosList[:, nodes-1] = myPosList[:, nodes-2]
        myPosList[:, 0] = myPos
    
    ###Hard Line Constraints###
    ALine = np.zeros((nodes, lenX*nodes))
    lLine = np.zeros((nodes, 1))
    uLine = np.zeros((nodes, 1))
    
    op = obstPos;
    obstVel = np.array(qo_i[d : 2*d])
    obstSpeed = math.sqrt(np.dot(obstVel, obstVel))
    for i in range(nodes):
        if len(prevTraj) != 0:
            myPos = myPosList[:, i]
        #print(myPos)
        op = op + obstVel * dt
        vec = myPos - op
        mag = math.sqrt(np.dot(vec, vec))
        unitVec = vec/mag
        
        pt = op + unitVec * setup['r_min']
        for j in range(d):
            ALine[i, j*nodes + i] = unitVec[j]
        lLine[i] = np.dot(unitVec, pt)
        uLine[i] = math.inf
        
    
    ###Delta Definition Constraints###
    ADelta = np.zeros((nodes, lenX*nodes))
    lDelta = np.zeros((nodes, 1))
    uDelta = np.zeros((nodes, 1))
    op = obstPos;
    
    factor = 0
    for i in range(nodes):
        if len(prevTraj) != 0:
            myPos = myPosList[:, i]
        op = op + obstVel * dt
        vec = myPos - op
        mag = math.sqrt(np.dot(vec, vec))
        unitVec = vec/mag
        
        pt = op + unitVec * setup['r_min2']
        for j in range(d):
            ADelta[i, j*nodes + i] = unitVec[j]
        ADelta[i, (lenX-2)*nodes + i] = -1
        lDelta[i] = np.dot(unitVec, pt)
        uDelta[i] = np.dot(unitVec, pt)
    
    ###Solve Optimization###
    ASetup = setup['ASetup']
    a1 = np.shape(ASetup)[0]
    a2 = np.shape(AEdge)[0]
    a3 = np.shape(ALine)[0]
    a4 = np.shape(ADelta)[0]
    height = a1 + a2 + a3 + a4
    A = np.zeros((height, lenX*nodes))
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
    traj = np.zeros((lenX, nodes))
    for i in range(lenX):
        traj[i, :] = np.transpose(sol[i*nodes : (i+1)*nodes])
    #print(traj)
    
    
    finalTraj = np.zeros((nodes, 33))
    finalTraj[:, 0] = dt
    for i in range(3):
        finalTraj[:, 8*i + 1] = traj[i, :]
        finalTraj[:, 8*i + 2] = traj[i + 3, :]
        for j in range(3, 9):
            finalTraj[:, 8*i + j] = np.gradient(finalTraj[:, 8*i + j - 1])
    finalTraj = finalTraj[1:, :]
    
    return finalTraj

#-------------------------------------------------------------------------------------