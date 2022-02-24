#First test of implementing MPC avoidance algorithm on drone (no csv)

import numpy as np
from pycrazyswarm import *
import uav_trajectory
import motioncapture
import time
import matplotlib.pyplot as plt
import ObstAvoid

#Defining time variable for keeping track of dt
lastObjTime = time.time()
lastDroneTime = time.time()

#Initializing array of previous positions (used in velocity calculations)
lastObjPos = np.array([0,0,0])
lastDronePos = np.array([0,0,0])

#Initializing state vectors
droneState = np.array([0, 0, 0, 0, 0, 0])
objState = np.array([0, 0, 0, 0, 0, 0])
desState = np.array([0.5, 0.5, 0, 0, 0, 0])

prevVels = np.zeros([5,3])


#Parameters to manipulate
sampleTime = 0.01  #amount of trajectory to be executed (in sec)
viconRate = 300 #Vicon sample rate

rate = 100.0    #rate of trajectory execution
Z = 0.3         #Z altitude
offset=np.array([0, 0, Z])

#Data Logging Setup
flightData = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=float)
trajData = []
timeIter = 0
runNum = 0

#Gets 3D state of object as numpy array
def getState3D(lastPos,pos,dTime):
    dPos = pos-lastPos
    vel = dPos.copy()/dTime
    state3D = np.array([pos[0],pos[1],pos[2],vel[0],vel[1],vel[2]])
    return state3D

#Gets 2D state of object as matlab double array
def getState(lastPos,pos,dTime):
    dPos = pos-lastPos
    vel = dPos.copy()/dTime
    state = matlab.double([pos[0].copy(),pos[1].copy(),vel[0].copy(),vel[1].copy()])
    state = eng.transpose(state)
    return state

def adjustState(state,solveTime):
    state = np.array(state)
    x = state[0,0]
    y = state[1,0]
    Vx = state[2,0]
    Vy = state[3,0]
    nx = x + (Vx*solveTime)
    ny = y + (Vy*solveTime)
    state = matlab.double([nx,ny,Vx,Vy])
    state = eng.transpose(state)
    return state

#Function SHOULD evaluate numpy trajectory the same way as uav_trajectory automatically
#evaluates csv's
def evaluate(myTraj, trajObj, tc):
    trajObj.polynomials = [uav_trajectory.Polynomial4D(row[0], row[1:9], row[9:17], row[17:25], row[25:33]) for row in myTraj]
    trajObj.duration = np.sum(myTraj[:, 0])
    return trajObj.eval(tc)
    
#Main control script
if __name__ == "__main__":

#SETUP (Runs once)#
    
    #Initialize drone
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    #cf.setParam('ctrlMel/kd_xy',0.15)
    #cf.setParam('ctrlMel/kp_xy',0.35)
    cf.setParam('stabilizer/controller',1)
    print(cf.getParam('stabilizer/controller'))

    #Call function to develop parameters for MPC
    setup = ObstAvoid.obst_avoid_setup()

    #Connect to mocap system
    mc = motioncapture.connect("vicon","192.168.1.119")

    #GET OBJECT AND DRONE STATE
    #Get one frame of data
    mc.waitForNextFrame()
    lt = time.time()
    obj = mc.rigidBodies["obst"]
    lastObjPos = obj.position.copy()
    drone = mc.rigidBodies["cf1"]
    lastDronePos = drone.position.copy()
    #Get next frame of data
    mc.waitForNextFrame()  
    obj = mc.rigidBodies["obst"]
    objPos = obj.position.copy()
    drone = mc.rigidBodies["cf1"]
    dronePos = drone.position.copy()
    dT = time.time()-lt

    #Get object state vector
    objState = getState(lastObjPos,objPos,dT)
    #Get drone state vector
    droneState = getState(lastDronePos,dronePos,dT)
    #Takeoff and wait for a second
    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)
    lt = time.time()
    
    solveTime = 0.1
    prevTraj = []
    
    #### Main avoidance loop ####
    try:
        while True:

            #Call next_traj function to analyze state and generate new trajectory
            newTraj = ObstAvoid.next_traj(setup, droneState, desState, objState, prevTraj)
            prevTraj = newTraj
            
            print('Run Time:',solveTime)
            traj = uav_trajectory.Trajectory()

            startTime = time.time()
            tc = 0
            while tc <= sampleTime:
                #Make sure dT is big enough to get good velocity measurements
                dT = time.time()-lt
                while dT <= 0.007:
                    dT = time.time()-lt
                
                #Get next frame of position data
                mc.waitForNextFrame()  
                obj = mc.rigidBodies["obst"]
                objPos = obj.position.copy()
                drone = mc.rigidBodies["cf1"]
                dronePos = drone.position.copy()

                #Calculate elapsed time since last position frame
                dT = time.time()-lt

                #Get object state vector
                objState = getState(lastObjPos,objPos,dT)
                #Get drone state vector
                droneState = getState(lastDronePos,dronePos,dT)
                
                #Save data for logging
                droneState3D = getState3D(lastDronePos,dronePos,dT)
                objState3D = getState3D(lastObjPos,objPos,dT)

                solveTime = time.time()-lt
    
                #Save previous position data for next iteration
                lt = time.time()
                lastDronePos = dronePos.copy()
                lastObjPos = objPos.copy()

                #Calculate current time (adding solve time to "skip" first few nodes)
                tc = (time.time()-startTime)

                #Interpolate and execute trajectory 
                e = evaluate(newTraj, traj, tc)   
                cf.cmdFullState(
                    e.pos + np.array(cf.initialPosition) + offset,
                    e.vel,
                    e.acc,
                    e.yaw,
                    e.omega)    
                timeHelper.sleepForRate(1000)

            runNum += 1
                
            

            #### END main avoidance loop ####
    except KeyboardInterrupt:
        print("E-Stop")
    finally:
        print("EmergencyStop Activated")
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.03, duration=Z+1.0)
        timeHelper.sleep(Z+2.0)