import pdb

import numpy as np


class Drone(object):
    numpy = __import__('numpy')
    sympy = __import__('sympy')
    osqp = __import__('osqp')
    scipy = __import__('scipy')
    pdb = __import__('pdb')

    def __init__(self, nodes, x0, desired_trajectory, **kwargs):
        """
        Initialize Attributes:
        """
        # Set Attributes:
        self.nodes = nodes
        self.reference_trajectory = desired_trajectory
        self.x0 = x0


    def initialize_optimization(self):
        """
        Initialize Optimization:
        """

        # Symbolic Variables:
        _x = self.sympy.symarray('_x', self.nodes)
        _dx = self.sympy.symarray('_dx', self.nodes)
        _y = self.sympy.symarray('_y', self.nodes)
        _dy = self.sympy.symarray('_dy', self.nodes)
        _f_x = self.sympy.symarray('f_x', self.nodes)
        _f_y = self.sympy.symarray('f_y', self.nodes)

        # Symbolic Variables representing Reference/Desired trajectory:
        _x_reference = self.sympy.symarray('x_reference', self.nodes)
        _dx_reference = self.sympy.symarray('dx_reference', self.nodes)
        _y_reference = self.sympy.symarray('y_reference', self.nodes)
        _dy_reference = self.sympy.symarray('dy_reference', self.nodes)

        # Design Variable Vector:
        _z = self.numpy.concatenate([_x, _dx, _f_x, _y, _dy, _f_y], axis=0)

        # Commonly used numbers:
        _number_of_states = 4
        _number_of_inputs = 2
        _number_of_design_variables = _number_of_states + _number_of_inputs
        _design_vector_length = len(_z)
        _design_vector_column_format = (self.nodes, _number_of_design_variables)

        # Reference/Desired Variable Vector:
        _reference_trajectory = self.numpy.concatenate([_x_reference, _dx_reference, _y_reference, _dy_reference], axis=0)

        # Objective Function:
        _objective_function = ((_x - _x_reference) ** 2 + (_y - _y_reference) ** 2 + (_dx - _dx_reference) ** 2 
                        + (_dy - _dy_reference) ** 2 + (_f_x ** 2 + _f_y ** 2))
        _objective_function = self.numpy.sum(_objective_function)

        # Compute Hessian:
        _H = [[_objective_function.diff(_axis_0).diff(_axis_1) for _axis_0 in _z] for _axis_1 in _z]

        # Compute Gradient:
        _f = [_objective_function.diff(_axis_0) for _axis_0 in _z]
        _f = [_f[_i].subs(_z[_i], 0) for _i in range(_design_vector_length)]

        # Convert H to self.numpy array:
        _H = self.numpy.asarray(_H, dtype=float)
        self.H = self.scipy.sparse.csc_matrix(_H)

        # Lambdify f:
        self.f = staticmethod(self.sympy.lambdify([_reference_trajectory], _f, 'numpy'))

        # States:
        _q = self.numpy.stack([_x, _y], axis=1)
        _dq = self.numpy.stack([_dx, _dy], axis=1)
        _u = self.numpy.stack([_f_x, _f_y], axis=1)

        # Planning Horizon:
        _planning_horizon = 1
        self.dt = _planning_horizon / (self.nodes - 1)

        # Equality Constraints:
        # Dynamics Constraints:
        _dynamics_position = _q[1:, :] - _q[:-1, :] - _dq[:-1, :] * self.dt
        _dynamics_velocity = _dq[1:, :] - _dq[:-1, :] - _u[:-1, :] * self.dt
        _collocation_constraints = self.numpy.stack((_dynamics_position, _dynamics_velocity), axis=0).flatten(order='F')
        _A_collocation, _b_collocation = self.sympy.linear_eq_to_matrix(_collocation_constraints, _z)
        _b_collocation = self.numpy.array(_b_collocation).astype(self.numpy.float64)
        self._b_collocation = _b_collocation.flatten(order='F')
        self._A_collocation = self.scipy.sparse.csc_matrix(_A_collocation, dtype=float)

        # Initial Conditions Constraints:
        _initial_condition = self.sympy.symarray('initial_condition', _number_of_states)
        _initial_conditions = self.numpy.concatenate(
                        (_q[0, :] - _initial_condition[:2], _dq[0, :] - _initial_condition[2:])).flatten(order='F')
        _A_initial_conditions, _b_initial_conditions = self.sympy.linear_eq_to_matrix(_initial_conditions, _z)
        self._A_initial_conditions = self.scipy.sparse.csc_matrix(_A_initial_conditions, dtype=float)
        self.initial_condition_constraint = staticmethod(self.sympy.lambdify([_initial_condition], _b_initial_conditions, 'numpy'))

        # Inequality Constraints:
        # Variable Bounds:
        # Pre-allocate Matrix:
        _design_variable_bound = self.numpy.zeros(_design_vector_column_format)
        # Defaults to Symmetric Bounds:
        position_bound = 10
        velocity_bound = 10
        force_bound = 10
        _design_variable_bound[:, 0] = position_bound
        _design_variable_bound[:, 1] = velocity_bound
        _design_variable_bound[:, 2] = force_bound
        _design_variable_bound[:, 3] = position_bound
        _design_variable_bound[:, 4] = velocity_bound
        _design_variable_bound[:, 5] = force_bound
        _design_variable_bound = _design_variable_bound.flatten(order='F')
        _design_variable_bound = _z[:] - _design_variable_bound
        _A_variable_bounds, _ = self.sympy.linear_eq_to_matrix(_design_variable_bound, _z)
        self._A_variable_bounds = self.scipy.sparse.csc_matrix(_A_variable_bounds, dtype=float)
        self._design_variable_bound = _design_variable_bound

        # Initialize QP:

        # Update Linear Terms:
        self.q = self.f(self.reference_trajectory)

        # Updates Equality Constraints:
        _A_equality = self.scipy.sparse.vstack([self._A_collocation, self._A_initial_conditions])
        _b_initial_conditions = self.initial_condition_constraint(self.x0)
        _lower_equality = self.numpy.concatenate([self._b_collocation, _b_initial_conditions], axis=0)
        _upper_equality = _lower_equality

        # Update Inequality Constraints:
        _A_inequality = self._A_variable_bounds
        _lower_inequality = self._design_variable_bound
        _upper_inequality = self._design_variable_bound

        # Combine Constraints:
        self.A = self.scipy.sparse.vstack([_A_equality, _A_inequality])
        self.l = self.numpy.concatenate([_lower_equality, _lower_inequality], axis=0)
        self.u = self.numpy.concatenate([_upper_equality, _upper_inequality], axis=0)

        # Create QP Object:
        self.qp = self.osqp.OSQP()
        self.qp.setup(self.H, self.q, self.A, self.l, self.u, warm_start=True)



    def update_optimization(self):
        """
        Update Optimization:
        """

        # Update Linear Terms:
        self.q = self.f(self.reference_trajectory)

        # Updates Equality Constraints:
        _A_equality = self.scipy.sparse.vstack([self._A_collocation, self._A_initial_conditions])
        _b_initial_conditions = self.initial_condition_constraint(self.x0)
        _lower_equality = self.numpy.concatenate([self._b_collocation, _b_initial_conditions], axis=0)
        _upper_equality = _lower_equality

        # Update Inequality Constraints:
        _A_inequality = self._A_variable_bounds
        _lower_inequality = self._design_variable_bound
        _upper_inequality = self._design_variable_bound

        # Combine Constraints:
        self.A = self.scipy.sparse.vstack([_A_equality, _A_inequality])
        self.l = self.numpy.concatenate([_lower_equality, _lower_inequality], axis=0)
        self.u = self.numpy.concatenate([_upper_equality, _upper_inequality], axis=0)

        self.qp.update()


    def generate_trajectory(self):
        """
        Generate Drone Trajectory:
        """

        # Run OSQP:
        self.solution = self.qp.solve(l=self.l, u=self.u)
        
        # Update x0: (Simulation Only)
        self.x0 = self.solution