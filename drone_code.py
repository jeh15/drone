# Import Libraries:
import numpy as np
from pycrazyswarm import *
import uav_trajectory
import motioncapture
import time
import matplotlib.pyplot as plt
import obstacle_avoidance

# HELPER FUNCTIONS:
#Gets 3D state of object as np array
def get_state(previous_positoion, position, dt):
    displacement = position - previous_positoion
    velocity = displacement.copy() / dt
    state = np.array([position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]])
    return state

#Function SHOULD evaluate numpy trajectory the same way as uav_trajectory automatically
#evaluates csv's
def evaluate(TrajectoryObject, trajectory, time):
    TrajectoryObject.polynomials = [uav_trajectory.Polynomial4D(row[0], row[1:9], row[9:17], row[17:25], row[25:33]) for row in trajectory]
    TrajectoryObject.duration = np.sum(trajectory[:, 0])
    return TrajectoryObject.eval(time)

#Defining time variable for keeping track of dt
previous_time_adversary = time.time()
previous_time_agent = time.time()

#Initializing array of previous positions (used in velocity calculations)
previous_position_adversary = np.array([0, 0, 0], dtype=float)
previous_position_agent = np.array([0, 0, 0], dtype=float)

#Initializing state vectors
state_agent = np.array([0, 0, 0, 0, 0, 0], dtype=float)
state_adversary = np.array([0, 0, 0, 0, 0, 0], dtype=float)
state_desired = np.array([0.5, 0.5, 0, 0, 0, 0], dtype=float)
previous_velocity = np.zeros([5, 3])

# Constants:
SAMPLE_TIME = 0.01      # amount of trajectory to be executed (in sec)
VICON_RATE = 300        # Vicon sample rate
RATE = 100.0            # rate of trajectory execution
ALTITUDE = 0.3          # Altitude
OFFSET = np.array([0, 0, ALTITUDE], dtype=float)

#Data Logging Setup
flight_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
trajectory_data = []
time_iteration = 0
run_number = 0
    
#Main control script
if __name__ == "__main__":

#SETUP (Runs once)#
    
    #Initialize drone
    Swarm = Crazyswarm()
    TimeHelper = Swarm.TimeHelper
    CrazyFly = Swarm.allcfs.crazyflies[0]
    CrazyFly.setParam('stabilizer/controller', 1)
    print(CrazyFly.getParam('stabilizer/controller'))

    #Call function to develop parameters for MPC
    setup = obstacle_avoidance.obstacle_avoidance_setup()

    #Connect to mocap system
    MotionCapture = motioncapture.connect("vicon","192.168.1.119")

    #GET OBJECT AND DRONE STATE
    #Get one frame of data
    MotionCapture.waitForNextFrame()
    previous_time = time.time()
    Adversary = MotionCapture.rigidBodies["obst"]
    previous_position_adversary = Adversary.position.copy()
    drone = MotionCapture.rigidBodies["cf1"]
    previous_position_agent = drone.position.copy()
    #Get next frame of data
    MotionCapture.waitForNextFrame()  
    Adversary = MotionCapture.rigidBodies["obst"]
    position_adversary = Adversary.position.copy()
    drone = MotionCapture.rigidBodies["cf1"]
    position_agent = drone.position.copy()
    dt = time.time() - previous_time

    #Get object state vector
    state_adversary = get_state(previous_position_adversary, position_adversary , dt)
    #Get drone state vector
    state_agent = get_state(previous_position_agent, position_agent, dt)
    #Takeoff and wait for a second
    CrazyFly.takeoff(targetHeight=ALTITUDE, duration=ALTITUDE+1.0)
    TimeHelper.sleep(ALTITUDE+2.0)
    previous_time = time.time()
    
    solve_time = 0.1
    previous_trajectory = []
    
    #### Main avoidance loop ####
    try:
        while True:

            #Call next_traj function to analyze state and generate new trajectory
            ## Is this a reference trajectory?
            trajectory = obstacle_avoidance.next_trajectory(setup, state_agent, state_desired, state_adversary, previous_trajectory)
            previous_trajectory = trajectory
            
            print('Run Time:',solveTime)
            traj = uav_trajectory.Trajectory()

            startTime = time.time()
            tc = 0
            while tc <= sampleTime:
                #Make sure dt is big enough to get good velocity measurements
                dt = time.time()-lt
                while dt <= 0.007:
                    dt = time.time()-lt
                
                #Get next frame of position data
                MotionCapture.waitForNextFrame()  
                Adversary = MotionCapture.rigidBodies["obst"]
                adversary_position = Adversary.position.copy()
                drone = MotionCapture.rigidBodies["cf1"]
                dronePos = drone.position.copy()

                #Calculate elapsed time since last position frame
                dt = time.time()-lt

                #Get object state vector
                objState = get_state(lastObjPos,adversary_position,dt)
                #Get drone state vector
                droneState = get_state(lastDronePos,dronePos,dt)

                solveTime = time.time()-lt
    
                #Save previous position data for next iteration
                lt = time.time()
                lastDronePos = dronePos.copy()
                lastObjPos = adversary_position.copy()

                #Calculate current time (adding solve time to "skip" first few nodes)
                tc = (time.time()-startTime)

                #Interpolate and execute trajectory 
                e = evaluate(newTraj, traj, tc)   
                CrazyFly.cmdFullState(
                    e.pos + np.array(CrazyFly.initialPosition) + offset,
                    e.vel,
                    e.acc,
                    e.yaw,
                    e.omega)    
                TimeHelper.sleepForRate(1000)

            run_number += 1

            #### END main avoidance loop ####
    except KeyboardInterrupt:
        print("E-Stop")
    finally:
        print("EmergencyStop Activated")
        CrazyFly.notifySetpointsStop()
        CrazyFly.land(targetHeight=0.03, duration=ALTITUDE+1.0)
        TimeHelper.sleep(ALTITUDE+2.0)
