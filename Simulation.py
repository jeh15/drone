import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import math
import random
import time
import scipy

# m = 1
# c = 1
# def f(state, t, Finterp, dt, m, c):
#     x = state[:, 0]
#     dx = state[:, 1]
#     myF = Finterp(t)
#     ddx = myF/m - c * dx
#     return np.hstack((ddx, dx))
#     
    
    
#F looks like [x1, x2, x3... ; y1, y2, y3... ; z1, z2, z3...]
def simulate_drone(T, dt, F, dx, x, m, c, numSamps):
    
#     r = ode(f).set_integrator('zvode', method='bdf')
#     r.set_initial_value([0, 0], t0).set_f_params(F, dt, m, c)
    
    ddt = T/(numSamps - 1)
#     print(x)
#     print(dx)
    for i in range(numSamps):
        t = (i - 1) * ddt
        ind = int(t/dt) + 1
        nextInd = ind + 1
        slope = (F[:, nextInd] - F[:, ind])/dt
        myF = F[:, ind] + slope * (t - dt * (ind - 1))
        
        ddx = myF/m - c * dx
        x = x + dx * ddt
        dx = dx + ddx * ddt
    state = np.zeros(6)
    state[0:3] = x[0:3]
    state[3:6] = dx[0:3]
    return state

def simulate_obst(T, qo_i, qd_i):
    relative = qo_i - qd_i
    Kp = 0.1
    Kd = 0.5
    F = -Kp * relative[0:3] - Kd * relative[3:]
    newPos = qo_i[0:3] + T * qo_i[3:]
    newVel = qo_i[3:] + T * F
    state = np.zeros(6)
#     print(newPos)
#     print(newVel)
    state[0:3] = newPos
    state[3:6] = newVel
    return state