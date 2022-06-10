import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import math
import random
import time

# numPoints = 50
# numDims = 1
# numSplinesPerDim = 2
# convex = False
# bound = False
# 
# points = np.zeros((numPoints, numDims + 1))
# for i in range(numPoints):
#     vec = 2 * np.random.rand(numDims)
#     #print(vec)
#     val = math.sqrt(np.dot(vec, vec)) + 1
#     points[i, :] = np.hstack((vec, val))
# #print(points)

def compress(hooks, p):
    ret = 0
    for i in range(np.size(hooks)):
        ret += p**i * hooks[i]
    return ret

#Input is [x1, x2... xn, y1]
#         [x1, x2... xn, y2]
#                ...
def splineFit(points, numSplinesPerDim, convex, bound):
    sz = np.shape(points)
    numPoints = sz[0]
    numDims = sz[1] - 1
    
    outputs = points[:, numDims]
    points = points[:, 0:numDims]
    
    mins = math.inf * np.ones((numDims))
    maxs = -math.inf * np.ones((numDims))
    for i in range(numPoints):
        for j in range(numDims):
            if points[i, j] < mins[j]:
                mins[j] = points[i, j]
            if points[i, j] > maxs[j]:
                maxs[j] = points[i, j]
    maxs = maxs + 0.1
    mins = mins - 0.1
    #print(maxs)
    #print(mins)
    
    for i in range(numPoints):
        for j in range(numDims):
            points[i, j] = points[i, j] - mins[j]
            if (maxs[j] != mins[j]):
                points[i, j] = points[i, j]/(maxs[j] - mins[j])
    
    numHooksPerDim = 1 + numSplinesPerDim
    hookGap = 1 / (numHooksPerDim - 1)
    
    h = numHooksPerDim
    n = numDims
    
    xPoints = np.zeros((h**n, n))
    clock = np.zeros((n))
    for i in range(h**n):
        #print(clock)
        xPoints[i, :] = np.multiply((hookGap * clock), (maxs - mins)) + mins
        ind = 0
        clock[ind] = clock[ind] + 1
        while clock[ind] == h and i < h**n - 1:
            clock[ind] = 0
            clock[ind + 1] = clock[ind + 1] + 1
            ind = ind + 1
    
    H = np.zeros((h**n, h**n))
    f = np.zeros((h**n, 1))
    for p in range(numPoints):
        hookList = np.zeros(numDims)
        cList = np.zeros(numDims)
        dList = np.zeros(numDims)
        for j in range(numDims):
            x = int((points[p, j]) / hookGap)
            if x == h - 1: #takes care of case when point is at max
                x = x - 1
            hookList[j] = x
            dList[j] = (points[p, j] - hookGap * hookList[j])/hookGap
        
        
        locList = np.zeros(2**numDims)
        prodList = np.zeros(2**numDims)
        binList = np.zeros(numDims)

        
        for j in range(2**numDims):
            prod = 1
            for k in range(numDims):
                if binList[k] == 1:
                    prod *= dList[k]
                else:
                    prod *= (1 - dList[k])
            prodList[j] = prod
            
            nList = hookList + binList
            locList[j] = compress(nList, h)
            
            binList[0] += 1
            if j < 2**numDims - 1:
                for k in range(numDims):
                    if binList[k] == 2:
                        binList[k] = 0
                        binList[k + 1] += 1
        
        #print(locList)
        for i in range(2**numDims):
            for j in range(2**numDims):
                hook1 = int(locList[i])
                #print(hook1)
                prod1 = prodList[i]
                hook2 = int(locList[j])
                #print(hook2)
                prod2 = prodList[j]
                H[hook1, hook2] += prod1 * prod2
        
        for i in range(2**numDims):
            hook1 = int(locList[i])
            prod1 = prodList[i]
            f[hook1, 0] += (-2 * outputs[p] * prod1)
    
    H *= 2
    
    boundA = np.identity(h**n)
    boundL = 0.001 * np.ones((h**n, 1))
    boundU = 0.999 * np.ones((h**n, 1))
    
    planeA = np.zeros(((n - 1)**2 * (h - 1)**n, h**n))
    root = np.zeros(n)
    root[0] = -1
    v = 0
    for i in range(h**n):
        ind = 0
        root[ind] += 1
        while root[ind] == h:
            root[ind] = 0
            root[ind + 1] += 1
            ind += 1
        
        if np.amax(root) < h - 1:
            for dim in range(n - 1):
                for adj in range(n):
                    if dim != adj:
                        a = h**dim
                        b = h**adj
                        planeA[v, i + a + b] = 1
                        planeA[v, i + b] = -1
                        planeA[v, i + a] = -1
                        planeA[v, i] = 1
                        v += 1
        
    planeL = np.zeros(((n - 1)**2 * (h - 1)**n, 1))
    planeU = np.zeros(((n - 1)**2 * (h - 1)**n, 1))
        
    convA = np.zeros((n * (h - 2), h**n))
    v = 0
    for dim in range(n):
        a = h**dim
        for i in range(h - 2):
            convA[v, a * (i + 2)] = 1
            convA[v, a * (i + 1)] = -2
            convA[v, a * i] = 1
            v += 1
    convL = -math.inf * np.ones((n * (h - 2), 1))
    convU = np.zeros((n * (h - 2), 1))
    
    A = np.zeros((1, h**n))
    l = np.zeros((1, 1))
    u = np.zeros((1, 1))
    
    if numDims > 1:
        A = np.vstack((A, planeA))
        l = np.vstack((l, planeL))
        u = np.vstack((u, planeU))
    
    if convex:
        A = np.vstack((A, convA))
        l = np.vstack((l, convL))
        u = np.vstack((u, convU))
        
    if bound:
        A = np.vstack((A, boundA))
        l = np.vstack((l, boundL))
        u = np.vstack((u, boundU))
        
    
    H = sparse.csc_matrix(H)
    A = sparse.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(H, f, A, l, u, alpha = 1.0, verbose = False)
    res = prob.solve()
    
    outputFinal = np.reshape(res.x, (h**n, 1))
    
    final = np.hstack((xPoints, outputFinal))
    #print(final)
    return final
    
        

# hooks = splineFit(points, numSplinesPerDim, convex, bound)
# for i in range(np.shape(hooks)[0]):
#     vec = hooks[i, 0:numDims]
#     error = hooks[i, numDims] - math.sqrt(np.dot(vec, vec))
#     print(error)