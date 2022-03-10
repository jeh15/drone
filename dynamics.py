# This function is to test the dynamics: WIP Make an autogenerate class for H, f, A, and b

import numpy
import sympy
import scipy
from scipy.sparse import csc_matrix

# Just for timing:
import time

nodes = 11
## Symbolic Formulation of H, f, A, and b matricies:

# Generate Design Variables:
x = sympy.symarray('x', nodes)
dx = sympy.symarray('dx', nodes)
y = sympy.symarray('y', nodes)
dy = sympy.symarray('dy', nodes)
f_x = sympy.symarray('f_x', nodes)
f_y = sympy.symarray('f_y', nodes)

# Generate Variables representing Reference/Desired trajectory:
x_desired = sympy.symarray('x_desired', nodes)
dx_desired = sympy.symarray('dx_desired', nodes)
y_desired = sympy.symarray('y_desired', nodes)
dy_desired = sympy.symarray('dy_desired', nodes)

# Design Variable Vector:
z = numpy.concatenate([x, dx, f_x, y, dy, f_y], axis=0)
# Commonly used numbers:
number_of_states = 4
number_of_inputs = 2
number_of_design_variables = number_of_states + number_of_inputs
design_vector_length = len(z)
design_vector_column_format = (nodes, number_of_design_variables)

# Reference/Desired Variable Vector:
desired_vector = numpy.concatenate([x_desired, dx_desired, y_desired, dy_desired], axis=0)

# Objective Function:
objective_function = (x - x_desired) ** 2 + (y - y_desired) ** 2 + (dx - dx_desired) ** 2 + (dy - dy_desired) ** 2 + (f_x ** 2 + f_y ** 2)
objective_function = numpy.sum(objective_function)

# Compute Hessian:
H = [[objective_function.diff(axis_0).diff(axis_1) for axis_0 in z] for axis_1 in z]

# Compute Gradient:
f = [objective_function.diff(axis_0) for axis_0 in z]
f = [f[i].subs(z[i], 0) for i in range(design_vector_length)]

# Convert H to numpy array:
H = numpy.asarray(H, dtype=float)
H = scipy.sparse.csc_matrix(H)

# Lambdify f:
# f = sympy.lambdify([desired_vector], sympy.SparseMatrix([f]), 'scipy')
f = sympy.lambdify([desired_vector], sympy.Matrix(f), 'numpy')

## Generate Constraint Matrix and Functions:
dt = 1.0    # Placeholder

# States:
q = numpy.stack([x, y], axis=1)
dq = numpy.stack([dx, dy], axis=1)
u = numpy.stack([f_x, f_y], axis=1)

# Equality Constraints:
# Dynamics Constraints: (Lets try to vectorize as much as possible)
constraints_dynamics_position = q[1:, :] - q[:-1, :] - dq[:-1, :] * dt
constraints_dynamics_velocity = dq[1:, :] - dq[:-1, :] - u[:-1, :] * dt
constraints_dynamics = numpy.row_stack((constraints_dynamics_position, constraints_dynamics_velocity))
constraints_dynamics = constraints_dynamics.flatten(order='F')
A_dynamics, b_dynamics = sympy.linear_eq_to_matrix(constraints_dynamics, z)
# These never have to be modified again:
A_dynamics = scipy.sparse.csc_matrix(A_dynamics, dtype=float)
# b_dynamics = scipy.sparse.csc_array(b_dynamics, dtype=float)

# Initial Conditions: (The b must be updated)
initial_condition = sympy.symarray('initial_condition', 4)
constraints_initial_conditions = numpy.row_stack((q[0, :] - initial_condition[:2], dq[0, :] - initial_condition[2:])).flatten(order='F')
A_initial_conditions, b_initial_conditions = sympy.linear_eq_to_matrix(constraints_initial_conditions, z)
A_initial_conditions = scipy.sparse.csc_matrix(A_initial_conditions, dtype=float)
b_initial_conditions = sympy.lambdify([initial_condition], b_initial_conditions, 'numpy')

# You need to get the results from the function first.
A_equality = scipy.sparse.vstack([A_dynamics, A_initial_conditions])
Dummy_Value = numpy.zeros((4,))
initial_condition_constraints = b_initial_conditions(Dummy_Value)
lower_equality = numpy.vstack([b_dynamics, initial_condition_constraints])
upper_equality = lower_equality

# Inequality Constraints:
# Variable Bounds:
# Pre-allocate Matrix:
design_variables_bound = numpy.zeros(design_vector_column_format)
# Symmetric Bounds:
position_bound = 2
velocity_bound = 10
force_bound = 10
design_variables_bound[:, 0] = position_bound
design_variables_bound[:, 1] = velocity_bound
design_variables_bound[:, 2] = force_bound
design_variables_bound[:, 3] = position_bound
design_variables_bound[:, 4] = velocity_bound
design_variables_bound[:, 5] = force_bound
design_variables_bound = design_variables_bound.flatten(order='F')
constraint_design_variables_lower_bound = -z[:] + design_variables_bound
constraint_design_variables_upper_bound = z[:] - design_variables_bound
# constraint_design_variables_bound = numpy.concatenate((constraint_design_variables_lower_bound, constraint_design_variables_upper_bound), axis=0)
# A_variable_bounds, b_variable_bounds = sympy.linear_eq_to_matrix(constraint_design_variables_bound, z)
# A_variable_bounds = scipy.sparse.csc_matrix(A_variable_bounds, dtype=float)

# Only solve relative to a single bounds:
constraint_design_variables_upper_bound = z[:] - design_variables_bound
A_variable_bounds, b_variable_bounds = sympy.linear_eq_to_matrix(constraint_design_variables_upper_bound, z)
A_variable_bounds = scipy.sparse.csc_matrix(A_variable_bounds, dtype=float)
lower_inequality = numpy.array([-design_variables_bound]).T
upper_inequality = numpy.array([design_variables_bound]).T

# Add Risk Constraints Here:

A_inequality = A_variable_bounds

# What needs updating: f, b_initial_conditions, (and risk when it gets added).

# Try to run the solver:
import osqp

qp = osqp.OSQP()

# Combine Constraints: (Don't forget to evaluate functions)
A = scipy.sparse.vstack([A_equality, A_inequality])
l = numpy.vstack([lower_equality, lower_inequality])
u = numpy.vstack([upper_equality, upper_inequality])

# Solve for updated f:
desired_trajectory = numpy.hstack([numpy.ones(nodes,), numpy.zeros(nodes,), numpy.ones(nodes,), numpy.zeros(nodes,)])
f_updated = f(desired_trajectory)

# Setup OSQP
qp.setup(H, f_updated, A, l, u, warm_start=True)

# Solve:
res = qp.solve()
solution = numpy.reshape(res.x, (nodes, number_of_design_variables), order='F')

# Plot Solution:
import matplotlib.pyplot as plt
plt.plot(solution[:, 0], solution[:, 3], marker='o')
plt.show()

# TO DO:
''' 
Remake problem such that we only use 1D array.
Remake problem to fit the format of l, u.
Remake workflow for updating f, b_initial_conditions, (and risk when it gets added).
Use a proper dt
'''