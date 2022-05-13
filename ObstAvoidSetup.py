import numpy as np
import math

def obst_avoid_setup():
    setup = {}
    setup['Th'] = 2.5;
    setup['Nodes'] = 11;
    setup['r_min'] = 0; #hard
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
    
    ###First-order dynamic constraints -- can be precomputed###
    
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
    
    ###Second-order dynamic constraints -- can be precomputed###
    
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
        
    setup['ASetup'] = np.vstack((ABound, ADyn1, ADyn2))
    setup['lSetup'] = np.vstack((lBound, lDyn1, lDyn2))
    setup['uSetup'] = np.vstack((uBound, uDyn1, uDyn2))
    
    return setup
    
    
    