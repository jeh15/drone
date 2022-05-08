class Drone_Risk(object):
    """
    Class Dependent Libraries:
    """

    numpy = __import__('numpy')
    sympy = __import__('sympy')
    osqp = __import__('osqp')
    scipy = __import__('scipy')
    integrate = __import__('scipy.integrate')
    interpolate = __import__('scipy.interpolate')
    pdb = __import__('pdb')

    def __init__(self, nodes, initial_condition, x0, desired_trajectory, **kwargs):
        """
        Initialize Attributes:
        """
        # Constant Attributes
        self.mass = 1.0

        # Set Attributes:
        self.nodes = nodes
        self.reference_trajectory = desired_trajectory
        self.x0 = x0
        self.initial_condition = initial_condition
        self._design_vector_column_format = 0
        self.control_horizon = self.numpy.zeros((3, 2))  # [f_x[:2]; f_y[:2]; f_z[:2]]
        self.spline_resolution = 5

        # Initialize Matrices: (Private)
        self._direction_vector = self.numpy.zeros((3, self.nodes))
        self.position = self.numpy.einsum("ij, i->ij", self.numpy.ones((3, self.nodes)), self.initial_condition[:3])
        self.velocity = self.numpy.einsum("ij, i->ij", self.numpy.ones((3, self.nodes)), self.initial_condition[3:])
        self.adversary_position = self.numpy.zeros((3, self.nodes))
        self.adversary_velocity = 0.0
        # risk_function format = [slope, y-intercept]
        self.risk_function = self.numpy.zeros((self.spline_resolution, 2))
        # Regression Matrices
        self._H_regression = self.numpy.zeros((self.spline_resolution+1, self.spline_resolution+1), dtype=float)
        self._H_regression_sparse = 0.0
        self._f_regression = self.numpy.zeros((self.spline_resolution+1,) , dtype=float)
        self._H_block = self.numpy.zeros((2, 2), dtype=float)
        self._f_block = self.numpy.zeros((2,), dtype=float)
        self.risk_regression_y = self.numpy.zeros((self.spline_resolution+1,), dtype=float)
        self.risk_regression_x = self.numpy.zeros((self.spline_resolution+1,), dtype=float)
        self._risk_weights = self.numpy.zeros((self.spline_resolution+1,), dtype=float)
        # risk sample format: [x values; y values]
        self.risk_sample = self.numpy.zeros((1, 1), dtype=float)

        # Initialize Attributues:
        self.simulation_solution = None

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

        # Slack Variables:
        _S = self.sympy.symarray('_S', self.nodes)

        # Symbolic Variables representing Reference/Desired trajectory:
        _x_reference = self.sympy.symarray('x_reference', self.nodes)
        _dx_reference = self.sympy.symarray('dx_reference', self.nodes)
        _y_reference = self.sympy.symarray('y_reference', self.nodes)
        _dy_reference = self.sympy.symarray('dy_reference', self.nodes)
        _z_reference = self.sympy.symarray('z_reference', self.nodes)
        _dz_reference = self.sympy.symarray('dz_reference', self.nodes)

        # Design Variable Vector:
        _X = self.numpy.concatenate([_x, _dx, _f_x, _y, _dy, _f_y, _z, _dz, _f_z, _S], axis=0)

        # Commonly used numbers:
        _number_of_states = 6
        _number_of_inputs = 3
        _number_of_slack = 1
        _number_of_design_variables = _number_of_states + _number_of_inputs + _number_of_slack
        _design_vector_length = len(_X)
        self._design_vector_column_format = (self.nodes, _number_of_design_variables)

        # Reference/Desired Variable Vector:
        _reference_trajectory = self.numpy.concatenate(
            [_x_reference, _dx_reference, _y_reference, _dy_reference, _z_reference, _dz_reference], axis=0)

        # Objective Function:
        _weight_distance = 10.0
        _weight_force = 1.0
        _objective_function = _weight_distance * ((_x - _x_reference) ** 2 + (_y - _y_reference) ** 2 + (_z - _z_reference) ** 2) \
                            + _weight_force * (_f_x ** 2 + _f_y ** 2) - _S

        _objective_function = self.numpy.sum(_objective_function)

        # Compute Hessian:
        _H = [[_objective_function.diff(_axis_0).diff(_axis_1) for _axis_0 in _X] for _axis_1 in _X]

        # Compute Gradient:
        _f = [_objective_function.diff(_axis_0) for _axis_0 in _X]

        # Speed this up?
        for _i in range(_design_vector_length):
            _eval = _f[_i]
            for _j in range(_design_vector_length):
                _eval = _eval.subs(_X[_j], 0)
            _f[_i] = _eval

        # OLD: Sympy Sucks and can't handle this...
        # _f = [_f[_i].subs(_X[_i], 0) for _i in range(_design_vector_length)]

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
        self._planning_horizon = 1.0
        self.dt = self._planning_horizon / (self.nodes - 1)
        self.tspan = [0.0, 2 * self.dt]
        self.t_eval = self.numpy.linspace(0, 2 * self.dt, 21)

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
        position_bound = 10.0
        velocity_bound = 10.0
        force_bound = 10.0
        slack_bound = 0.0
        _design_variable_bound[:, 0] = position_bound
        _design_variable_bound[:, 1] = velocity_bound
        _design_variable_bound[:, 2] = force_bound
        _design_variable_bound[:, 3] = position_bound
        _design_variable_bound[:, 4] = velocity_bound
        _design_variable_bound[:, 5] = force_bound
        _design_variable_bound[:, 6] = position_bound
        _design_variable_bound[:, 7] = velocity_bound
        _design_variable_bound[:, 8] = force_bound
        _design_variable_bound[:, 9] = slack_bound
        self._design_variable_bound = _design_variable_bound.flatten(order='F')
        _design_variable_bound = _X[:] - self._design_variable_bound
        _A_variable_bounds, _ = self.sympy.linear_eq_to_matrix(_design_variable_bound, _X)
        self._A_variable_bounds = self.scipy.sparse.csc_matrix(_A_variable_bounds, dtype=float)

        # Risk Constraint:
        # direction vector data format: [nx; ny; nz]
        self._direction_vector[:, :] = self.adversary_position - self.position
        _adversary_position = self.sympy.symarray('_adversary_position', self.adversary_position.shape)
        _direction_vector = self.sympy.symarray('_direction_vector', self._direction_vector.shape)
        _delta = _direction_vector[0, :] * (_x - _adversary_position[0, :]) \
                               + _direction_vector[1, :] * (_y - _adversary_position[1, :]) \
                               + _direction_vector[2, :] * (_z - _adversary_position[2, :])

        _risk_slope = self.sympy.symarray('_risk_slope', self.spline_resolution)
        _risk_intercept = self.sympy.symarray('_risk_intercept', self.spline_resolution)

        # Pre-allocate symbolic matrix:
        _rfun = self.numpy.empty(self.spline_resolution, dtype=object)
        for i in range(self.spline_resolution):
            _rfun[i] = _S - (_risk_slope[i] * _delta + _risk_intercept[i])

        # Risk Constraint:
        _A_risk = []
        _b_risk = []
        for i in range(self.spline_resolution):
            _A_intermediate, _b_intermediate = self.sympy.linear_eq_to_matrix(_rfun[i], _X)
            _A_risk.append(_A_intermediate)
            _b_risk.append(_b_intermediate)

        _A_risk = self.sympy.Matrix(_A_risk)
        _b_risk = self.sympy.Matrix(_b_risk)

        # _A_risk and _b_risk input format:
        _replacements = self.numpy.concatenate((_direction_vector.flatten(order='C'),
                                                _adversary_position.flatten(order='C'),
                                                _risk_slope,
                                                _risk_intercept), axis=0)

        self._A_risk = self.sympy.lambdify([_replacements], _A_risk, 'numpy')
        self._b_risk = self.sympy.lambdify([_replacements], _b_risk, 'numpy')

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
                                                self.adversary_position.flatten(order='C'),
                                                self.risk_function[:, 0],
                                                self.risk_function[:, 1]), axis=0)

        _A_risk = self.scipy.sparse.csc_matrix(self._A_risk(_replacements))
        _upper_risk = (self._b_risk(_replacements)).flatten(order='F')
        self._lower_risk = (-self.numpy.inf * self.numpy.ones(self.numpy.shape(_upper_risk))).flatten(order='F')

        _A_inequality = self.scipy.sparse.vstack((self._A_variable_bounds, _A_risk))

        self._lower_design_variable_bound = -self._design_variable_bound
        self._lower_design_variable_bound[-self.nodes:] = -self.numpy.inf

        _lower_inequality = self.numpy.concatenate((self._lower_design_variable_bound, self._lower_risk), axis=0)
        _upper_inequality = self.numpy.concatenate((self._design_variable_bound, _upper_risk), axis=0)

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
        self.get_adversary_info()  # <-- Useful for when we have vicon support

        # Truncate already executed trajectory: (Make control horizon a variable)
        self.position[:, :-2] = self.position[:, 2:]
        self.position[:, -2:] = self.numpy.einsum('i,ij->ij', self.position[:, -1], self.numpy.ones(self.position[:, -2:].shape))
        self._direction_vector[:, :] = self.adversary_position - self.position
        _replacements = self.numpy.concatenate((self._direction_vector.flatten(order='C'),
                                                self.adversary_position.flatten(order='C'),
                                                self.risk_function[:, 0],
                                                self.risk_function[:, 1]), axis=0)
        _A_risk = self.scipy.sparse.csc_matrix(self._A_risk(_replacements))
        _upper_risk = (self._b_risk(_replacements)).flatten(order='F')
        _A_inequality = self.scipy.sparse.vstack((self._A_variable_bounds, _A_risk))
        _lower_inequality = self.numpy.concatenate((self._lower_design_variable_bound, self._lower_risk), axis=0)
        _upper_inequality = self.numpy.concatenate((self._design_variable_bound, _upper_risk), axis=0)

        # Combine Constraints: (A Needs to be updated for Half-space constraints)
        self.A = self.scipy.sparse.vstack((_A_equality, _A_inequality))
        self.l = self.numpy.concatenate((_lower_equality, _lower_inequality), axis=0)
        self.u = self.numpy.concatenate((_upper_equality, _upper_inequality), axis=0)

        self.qp.update(Ax=self.A.data, l=self.l, u=self.u)

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

        # Set Temporary Variable to hold Solution formatted Column-wise:
        # _temp data format: [x, dx, f_x, y, dy, f_y, z, dz, f_z]
        _temp = self.solution.x.reshape(self._design_vector_column_format, order='F')

        # Set Next Control Trajectory:
        self.control_horizon[:, :] = self.numpy.array((_temp[:2, 2], _temp[:2, 5], _temp[:2, 8]), dtype=float)
        self.control_function = self.scipy.interpolate.interp1d(self.tspan, self.control_horizon)

        # Update x0: (Simulation Only)
        self.position[:, :] = self.numpy.vstack((_temp[:, 0], _temp[:, 3], _temp[:, 6]))
        self.velocity[:, :] = self.numpy.vstack((_temp[:, 1], _temp[:, 4], _temp[:, 7]))
        self.x0 = self.solution.x

    def simulate(self):
        # solve_ivp(fun=lambda t, y: fun(t, y, *args), ...)
        self.simulation_solution = self.scipy.integrate.solve_ivp(lambda t, y: self.ode_func(t, y, self), self.tspan,
                                                                  self.initial_condition, method='RK45',
                                                                  t_eval=self.t_eval, vectorized=True)
        # initial_conditions Data format: [x, y, z, dx, dy, dz]
        self.initial_condition[:] = self.simulation_solution.y[:, -1]

    def get_failure_probability_function(self):
        # Update H and f matrices for risk regression:
        self.get_objective()
        # Update Problem Matrices:
        _H = self._H_regression.T + self._H_regression
        self.numpy.fill_diagonal(_H, self.numpy.diag(self._H_regression))
        self._H_regression_sparse = self.scipy.sparse.csc_matrix(_H)
        self.risk_regression.update(Px=self.scipy.sparse.triu(self._H_regression_sparse).data, q=self._f_regression)
        # Solve Optimization:
        self.risk_regression_solution = self.risk_regression.solve()
        self.risk_regression_y = self.risk_regression_solution.x[:]

    def initialize_risk_regression(self):
        # Update H and f matrices for risk regression:
        self.get_objective()
        # Make Triangular Matrix Full and Convert to CSC Format:
        _H = self._H_regression.T + self._H_regression
        self.numpy.fill_diagonal(_H, self.numpy.diag(self._H_regression))
        self._H_regression_sparse = self.scipy.sparse.csc_matrix(_H)
        # Get A Matrix, lower, and upper bounds:
        _A = self.scipy.sparse.eye(self.spline_resolution+1, format='csc')
        _l = self.numpy.zeros((self.spline_resolution+1,), dtype=float)
        _u = self.numpy.ones((self.spline_resolution+1,), dtype=float)
        # Initialize Failure Probability Regression:
        self.risk_regression = self.osqp.OSQP()
        # Setup Problem:
        self.risk_regression.setup(self._H_regression_sparse, self._f_regression, _A, _l, _u, warm_start=True)

    def get_objective(self):
        # Data Point Locations:
        _xd = self.risk_sample[0, :]
        _yd = self.risk_sample[1, :]
        self.risk_regression_x[:] = self.numpy.linspace(_xd[0], _xd[-1], self.spline_resolution+1)
        _x = self.risk_regression_x

        # Reset Matrices:
        self._H_regression[:, :] = 0.0
        self._f_regression[:] = 0.0

        # Find the weights of each spline:
        _risk_weights = self.numpy.zeros((self.spline_resolution,))
        for _k in range(self.spline_resolution):
            _risk_weights[_k] = self.numpy.sum((_xd[:] > _x[_k]) & (_xd[:] <= _x[_k+1]))

        for _k in range(self.spline_resolution+1):
            if _k == 0:
                self._risk_weights[_k] = _risk_weights[0]
            elif _k == self.spline_resolution:
                self._risk_weights[_k] = _risk_weights[-1]
            else:
                self._risk_weights[_k] = _risk_weights[_k-1] + _risk_weights[_k]

        # Compute Hessian and Gradient:
        j = 0
        for i in range(len(_xd)):
            if _xd[i] >= _x[j+1]:
                self._H_regression[j:j+2, j:j+2] = self._H_regression[j:j+2, j:j+2] + self._risk_weights[j] * self._H_block
                self._f_regression[j:j+2] = self._f_regression[j:j+2] + self._risk_weights[j] * self._f_block
                j = j + 1
                self._H_block[:, :] = 0.0
                self._f_block[:] = 0.0

            if _xd[i] == _x[j]:
                self._H_block[0, 0] = 2.0
                self._f_block[0] = -2.0 * _yd[i]
            else:
                span = _x[j] - _x[j+1]
                span_sq = span ** 2
                upper_span = _xd[i] - _x[j + 1]
                lower_span = _xd[i] - _x[j]

                self._H_block[0, 0] = self._H_block[0, 0] + 2.0 * upper_span ** 2 / span_sq
                self._H_block[1, 0] = self._H_block[1, 0] + -2.0 * lower_span * upper_span / span_sq
                self._H_block[1, 1] = self._H_block[1, 1] + 2.0 * lower_span ** 2 / span_sq

                self._f_block[0] = self._f_block[0] + -2.0 * _yd[i] * upper_span / span
                self._f_block[1] = self._f_block[1] + 2.0 * _yd[i] * lower_span / span

        # Add End Point:
        self._H_regression[-1, -1] = self._H_regression[-1, -1] + self._risk_weights[-1] * self._H_block[0, 0]
        self._f_regression[-1] = self._f_regression[-1] + self._risk_weights[-1] * self._f_block[0]

        # self._H_regression[-1, -1] = self._H_regression[-1, -1] + self._risk_weights[-1] * 2.0
        # self._f_regression[-1] = self._f_regression[-1] - 2.0 * self._risk_weights[-1] * _yd[-1]

        # self.pdb.set_trace()

        # Make Matrix Upper Triangular:
        self._H_regression[:, :] = self._H_regression.T


    @staticmethod
    def ode_func(t, y, self):
        u = self.control_function(t)
        dx = y[3]
        ddx = u[0] / self.mass
        dy = y[4]
        ddy = u[1] / self.mass
        dz = y[5]
        ddz = u[2] / self.mass
        return [dx, dy, dz, ddx, ddy, ddz]

