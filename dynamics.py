import numpy
import sympy
import scipy
from scipy.sparse import csc_matrix

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
z = numpy.concatenate([x, dx, y, dy, f_x, f_y], axis=0)

# Reference/Desired Variable Vector:
desired_vector = numpy.concatenate([x_desired, dx_desired, y_desired, dy_desired], axis=0)

# Objective Function:
objective_function = (x - x_desired) ** 2 + (y - y_desired) ** 2 + (dx - dx_desired) ** 2 + (dy - dy_desired) ** 2 + (f_x ** 2 + f_y ** 2)
objective_function = numpy.sum(objective_function)

# Compute Hessian:
H = [[objective_function.diff(axis_0).diff(axis_1) for axis_0 in z] for axis_1 in z]

# Compute Gradient:
f = [objective_function.diff(axis_0) for axis_0 in z]
f = [f[i].subs(z[i], 0) for i in range(len(z))]

change is here

# Convert H to numpy array:
H = numpy.asarray(H, dtype=float)
H = scipy.sparse.csc_matrix(H)

# Lambdify f:
f = sympy.lambdify([desired_vector], sympy.SparseMatrix([f]), 'scipy')

import inspect
print(inspect.getsource(f))