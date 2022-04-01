import numpy
import sympy
import osqp
import scipy
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt
import pdb

import drone

nodes = 21
number_of_states = 5
x0 = numpy.zeros((nodes * number_of_states, 1))
desired_trajectory = numpy.concatenate((numpy.ones((nodes,)), numpy.zeros((nodes,)), numpy.ones((nodes,)), numpy.zeros((nodes,))), axis=0)

agent = drone.Drone(nodes, x0, desired_trajectory)

# Initialize Optimization:
agent.initialize_optimization()

# Update Constraints:
agent.update_optimization()

# Generate Trajectory:
agent.generate_trajectory()
