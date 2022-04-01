import Drone
import numpy

nodes = 21
number_of_states = 5
x0 = numpy.zeros((nodes * number_of_states, 1))
desired_trajectory = numpy.concatenate((numpy.ones((nodes,)), numpy.zeros((nodes,)), numpy.ones((nodes,)), numpy.zeros((nodes,))), axis=0)

drone = Drone(nodes, x0, desired_trajectory)