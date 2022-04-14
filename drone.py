class Drone(object):
    """
    Class Dependent Libraries:
    """

    numpy = __import__('numpy')
    sympy = __import__('sympy')
    osqp = __import__('osqp')
    scipy = __import__('scipy')
    pdb = __import__('pdb')

    def __init__(self, nodes, initial_condition, x0, desired_trajectory, **kwargs):
        """
        Initialize Attributes:
        """

        # Set Attributes:
        self.nodes = nodes
        self.reference_trajectory = desired_trajectory
        self.x0 = x0
        self.initial_condition = initial_condition
        self._design_vector_column_format = 0

        # Initialize Matrices: (Private)
        self._direction_vector = self.numpy.zeros((3, self.nodes))
        self.position = self.numpy.zeros((3, self.nodes))  # INITIALIZE WITH x0
        self.velocity = self.numpy.zeros((3, self.nodes))  # INITIALIZE WITH x0
        self.adversary_position = self.numpy.zeros((3, self.nodes))
        self.adversary_velocity = 0.0

    def initialize_optimization(self):
        """
        Initialize Optimization:
        """

        # Symbolic Variables:
        _x = self.sympy.symarray('_x', self.nodes)
        _dx = self.sympy.symarray('_dx', self.nodes)
        _y = self.sympy.symarray('_y', self.nodes)
        _dy = self.sympy.symarray('_dy', self.nodes)
        _z = self.sympy.symarray('_z', self.nodes)
        _dz = self.sympy.symarray('_dz', self.nodes)
        _f_x = self.sympy.symarray('f_x', self.nodes)
        _f_y = self.sympy.symarray('f_y', self.nodes)
        _f_z = self.sympy.symarray('f_z', self.nodes)

        # Symbolic Variables representing Reference/Desired trajectory:
        _x_reference = self.sympy.symarray('x_reference', self.nodes)
        _dx_reference = self.sympy.symarray('dx_reference', self.nodes)
        _y_reference = self.sympy.symarray('y_reference', self.nodes)
        _dy_reference = self.sympy.symarray('dy_reference', self.nodes)
        _z_reference = self.sympy.symarray('z_reference', self.nodes)
        _dz_reference = self.sympy.symarray('dz_reference', self.nodes)

        # Design Variable Vector:
        _X = self.numpy.concatenate([_x, _dx, _f_x, _y, _dy, _f_y, _z, _dz, _f_z], axis=0)

        # Commonly used numbers:
        _number_of_states = 6
        _number_of_inputs = 3
        _number_of_design_variables = _number_of_states + _number_of_inputs
        _design_vector_length = len(_X)
        self._design_vector_column_format = (self.nodes, _number_of_design_variables)

        # Reference/Desired Variable Vector:
        _reference_trajectory = self.numpy.concatenate(
            [_x_reference, _dx_reference, _y_reference, _dy_reference, _z_reference, _dz_reference], axis=0)

        # Objective Function:
        _objective_function = ((_x - _x_reference) ** 2 + (_y - _y_reference) ** 2 + (_dx - _dx_reference) ** 2
                               + (_dy - _dy_reference) ** 2 + (_f_x ** 2 + _f_y ** 2))

        _objective_function = self.numpy.sum(_objective_function)

        # Compute Hessian:
        _H = [[_objective_function.diff(_axis_0).diff(_axis_1) for _axis_0 in _X] for _axis_1 in _X]

        # Compute Gradient:
        _f = [_objective_function.diff(_axis_0) for _axis_0 in _X]
        _f = [_f[_i].subs(_X[_i], 0) for _i in range(_design_vector_length)]

        # Convert H to self.numpy array:
        _H = self.numpy.asarray(_H, dtype=float)
        self.H = self.scipy.sparse.csc_matrix(_H)

        # Lambdify f:
        self.f = self.sympy.lambdify([_reference_trajectory], _f, 'numpy')

        # States:
        _q = self.numpy.stack([_x, _y, _z], axis=1)
        _dq = self.numpy.stack([_dx, _dy, _dz], axis=1)
        _u = self.numpy.stack([_f_x, _f_y, _f_z], axis=1)

        # Planning Horizon:
        _planning_horizon = 1
        self.dt = _planning_horizon / (self.nodes - 1)

        """
        Equality Constraints:
        """

        # Dynamic Collocation Constraints: (DOES NOT ACCOUNT FOR GRAVITY)
        _dynamics_position = _q[1:, :] - _q[:-1, :] - _dq[:-1, :] * self.dt
        _dynamics_velocity = _dq[1:, :] - _dq[:-1, :] - _u[:-1, :] * self.dt
        _collocation_constraints = self.numpy.stack((_dynamics_position, _dynamics_velocity), axis=0).flatten(order='F')
        _A_collocation, _b_collocation = self.sympy.linear_eq_to_matrix(_collocation_constraints, _X)
        _b_collocation = self.numpy.array(_b_collocation).astype(self.numpy.float64)
        self._b_collocation = _b_collocation.flatten(order='F')
        self._A_collocation = self.scipy.sparse.csc_matrix(_A_collocation, dtype=float)

        # Initial Conditions Constraints:
        _initial_condition = self.sympy.symarray('initial_condition', _number_of_states)
        _initial_conditions = self.numpy.concatenate(
            (_q[0, :] - _initial_condition[:3], _dq[0, :] - _initial_condition[3:])).flatten(order='F')
        _A_initial_conditions, _b_initial_conditions = self.sympy.linear_eq_to_matrix(_initial_conditions, _X)
        _b_initial_conditions = _b_initial_conditions.reshape(1, len(_b_initial_conditions))
        self._A_initial_conditions = self.scipy.sparse.csc_matrix(_A_initial_conditions, dtype=float)
        self.initial_condition_constraint = self.sympy.lambdify([_initial_condition], _b_initial_conditions, 'numpy')

        """
        Inequality Constraints:
        """

        # Design Variable Bounds:
        # Pre-allocate Matrix:
        _design_variable_bound = self.numpy.zeros(self._design_vector_column_format)
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
        _design_variable_bound[:, 6] = position_bound
        _design_variable_bound[:, 7] = velocity_bound
        _design_variable_bound[:, 8] = force_bound
        self._design_variable_bound = _design_variable_bound.flatten(order='F')
        _design_variable_bound = _X[:] - self._design_variable_bound
        _A_variable_bounds, _ = self.sympy.linear_eq_to_matrix(_design_variable_bound, _X)
        self._A_variable_bounds = self.scipy.sparse.csc_matrix(_A_variable_bounds, dtype=float)

        # Half-Space Constraint: (TO DO) (Only on un-executed trajectory)
        # direction vector data format: [nx; ny; nz]
        self._direction_vector[:, :] = self.adversary_position - self.position
        _adversary_position = self.sympy.symarray('_adversary_position', self.adversary_position.shape)
        _direction_vector = self.sympy.symarray('_direction_vector', self._direction_vector.shape)
        _hyperplane_constraint = _direction_vector[0, :] * (_x - _adversary_position[0, :]) \
                               + _direction_vector[1, :] * (_y - _adversary_position[1, :]) \
                               + _direction_vector[2, :] * (_z - _adversary_position[2, :])

        _A_halfspace, _b_halfspace = self.sympy.linear_eq_to_matrix(_hyperplane_constraint, _X)

        _replacements = self.numpy.concatenate((_direction_vector.flatten(order='C'),
                                                _adversary_position.flatten(order='C')), axis=0)
        # We could initialize all of these... (TO DO PREALLOCATE ALL ARRAY OPERATIONS)
        self._A_halfspace = self.sympy.lambdify([_replacements], _A_halfspace, 'numpy')
        self._b_halfspace = self.sympy.lambdify([_replacements], _b_halfspace, 'numpy')

        """
        Initialize QP:
        """

        # Update Linear Terms:
        self.q = self.numpy.asarray(self.f(self.reference_trajectory), dtype=float)

        # Updates Equality Constraints:
        _A_equality = self.scipy.sparse.vstack((self._A_collocation, self._A_initial_conditions))
        _b_initial_conditions = self.initial_condition_constraint(self.initial_condition)
        _lower_equality = self.numpy.concatenate((self._b_collocation, _b_initial_conditions.flatten(order='F')),
                                                 axis=0)
        _upper_equality = _lower_equality

        # Update Inequality Constraints:
        self._direction_vector[:, :] = self.adversary_position - self.position
        _replacements = self.numpy.concatenate((self._direction_vector.flatten(order='C'),
                                                self.adversary_position.flatten(order='C')), axis=0)
        _A_halfspace = self.scipy.sparse.csc_matrix(self._A_halfspace(_replacements))
        _upper_halfspace = (self._b_halfspace(_replacements)).flatten(order='F')
        self._lower_halfspace = (-self.numpy.inf * self.numpy.ones(self.numpy.shape(_upper_halfspace))).flatten(order='F')
        _A_inequality = self.scipy.sparse.vstack((self._A_variable_bounds, _A_halfspace))
        _lower_inequality = self.numpy.concatenate((-self._design_variable_bound, self._lower_halfspace), axis=0)
        _upper_inequality = self.numpy.concatenate((self._design_variable_bound, _upper_halfspace), axis=0)

        # Combine Constraints:
        self.A = self.scipy.sparse.vstack((_A_equality, _A_inequality))
        self.l = self.numpy.concatenate((_lower_equality, _lower_inequality), axis=0)
        self.u = self.numpy.concatenate((_upper_equality, _upper_inequality), axis=0)

        # Create QP Object:
        self.qp = self.osqp.OSQP()
        self.qp.setup(self.H, self.q, self.A, self.l, self.u, warm_start=True)

    def update_optimization(self):
        """
        Update Optimization:
        """

        # Update Linear Terms:
        self.q = self.numpy.asarray(self.f(self.reference_trajectory), dtype=float)

        # Updates Equality Constraints:
        _A_equality = self.scipy.sparse.vstack((self._A_collocation, self._A_initial_conditions))
        _b_initial_conditions = self.initial_condition_constraint(self.initial_condition)
        _lower_equality = self.numpy.concatenate((self._b_collocation, _b_initial_conditions.flatten(order='F')),
                                                 axis=0)
        _upper_equality = _lower_equality

        # Update Inequality Constraints:
        self.get_adversary_info()
        self._direction_vector[:, :] = self.adversary_position - self.position
        _replacements = self.numpy.concatenate((self._direction_vector.flatten(order='C'),
                                                self.adversary_position.flatten(order='C')), axis=0)
        _A_halfspace = self.scipy.sparse.csc_matrix(self._A_halfspace(_replacements))
        _upper_halfspace = (self._b_halfspace(_replacements)).flatten(order='F')
        _A_inequality = self.scipy.sparse.vstack((self._A_variable_bounds, _A_halfspace))
        _lower_inequality = self.numpy.concatenate((-self._design_variable_bound, self._lower_halfspace), axis=0)
        _upper_inequality = self.numpy.concatenate((self._design_variable_bound, _upper_halfspace), axis=0)

        # Combine Constraints: (A Needs to be updated for Half-space constraints)
        self.A = self.scipy.sparse.vstack((_A_equality, _A_inequality))
        self.l = self.numpy.concatenate((_lower_equality, _lower_inequality), axis=0)
        self.u = self.numpy.concatenate((_upper_equality, _upper_inequality), axis=0)

        self.qp.update(Ax=self.A.data, l=self.l, u=self.u)

    def get_halfspace_constraints(self):
        """
        Generate Half-Space Constraints:
        """
        self.get_adversary_info()
        # direction vector data format: [nx; ny; nz]
        self._direction_vector[:, :] = self.adversary_position - self.position

    def get_adversary_info(self):
        """
        Get current adversary information from VICON.
        Predict N nodes into the future of adversary position.
        """
        # adversary_position data format : [x; y; z]
        self.adversary_position[:, :] = self.numpy.zeros((3, self.nodes))

    def generate_trajectory(self):
        """
        Generate Drone Trajectory:
        """

        # Run OSQP:
        self.solution = self.qp.solve()

        # Reshape and Set Initial Condition: (Initial Condition Order x, y, z, dx, dy, dz)
        _temp = self.solution.x.reshape(self._design_vector_column_format, order='F')
        # Position and Velocity Data Format = [x; y; z] / [dx; dy; dz]
        self.position[:, :] = self.numpy.vstack((_temp[:, 0], _temp[:, 3], _temp[:, 6]))
        self.velocity[:, :] = self.numpy.vstack((_temp[:, 1], _temp[:, 4], _temp[:, 7]))
        _temp = self.numpy.array((_temp[-1, 0], _temp[-1, 3], _temp[-1, 6], _temp[-1, 1], _temp[-1, 4], _temp[-1, 7]))
        self.initial_condition = _temp

        # Update x0: (Simulation Only)
        self.x0 = self.solution.x
