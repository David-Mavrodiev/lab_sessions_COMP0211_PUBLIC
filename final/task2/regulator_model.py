import numpy as np
from scipy.linalg import solve_discrete_are

class RegulatorModel:
    def __init__(self, N, q, m, n):
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.R = None
        self.P = None  # Terminal weight matrix, computed if needed
        self.N = N
        self.q = q  # Output dimension
        self.m = m  # Input dimension
        self.n = n  # State dimension

    def compute_H_and_F_without_P(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H and F matrices without terminal cost P
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar
        F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))
        return H, F

    def compute_H_and_F_with_P(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H and F matrices, incorporating terminal cost P
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar
        F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))
        
        # Add terminal cost for the final state
        #H[-self.q:, -self.q:] += self.P
        return H, F

    def propagation_model_regulator_fixed_std(self):
        S_bar = np.zeros((self.N * self.q, self.N * self.m))
        T_bar = np.zeros((self.N * self.q, self.n))
        Q_bar = np.zeros((self.N * self.q, self.N * self.q))
        R_bar = np.zeros((self.N * self.m, self.N * self.m))

        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                S_bar[(k-1)*self.q:k*self.q, (k-j)*self.m:(k-j+1)*self.m] = np.dot(np.dot(self.C, np.linalg.matrix_power(self.A, j-1)), self.B)

            T_bar[(k-1)*self.q:k*self.q, :self.n] = np.dot(self.C, np.linalg.matrix_power(self.A, k))
            #Q_bar[(k-1)*self.q:k*self.q, (k-1)*self.q:k*self.q] = self.Q
            if k == self.N and self.P is not None:
                Q_bar[(k-1)*self.q:k*self.q, (k-1)*self.q:k*self.q] = self.P  # Terminal state weight
            else:
                Q_bar[(k-1)*self.q:k*self.q, (k-1)*self.q:k*self.q] = self.Q
            R_bar[(k-1)*self.m:k*self.m, (k-1)*self.m:k*self.m] = self.R

        return S_bar, T_bar, Q_bar, R_bar
    
    def computeTerminalWeightMatrix(self):
        """
        Compute the terminal weight matrix P using the Discrete Algebraic Riccati Equation.
        """
        if self.A is None or self.B is None or self.Q is None or self.R is None:
            raise ValueError("A, B, Q, and R must be set before computing the terminal weight matrix P.")

        try:
            # Solve DARE to compute P
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            self.P = np.array(P, dtype=np.float64)  # Ensure P is a numeric matrix
        except np.linalg.LinAlgError as e:
            raise ValueError("Failed to compute terminal weight matrix P: " + str(e))


    def updateSystemMatrices(self, sim, cur_x, cur_u):
        if cur_x is None or cur_u is None:
            raise ValueError("State and control inputs must be provided for linearization.")
        
        # Extract required values from cur_x and cur_u
        v0 = cur_u[0]  # Current velocity (first control input)
        theta0 = cur_x[2]  # Current orientation (theta)
        delta_t = sim.GetTimeStep()  # Time step from simulator

        # Define the A and B matrices based on linearized dynamics
        # System matrices
        self.A = np.array([
            [1, 0, -v0 * delta_t * np.sin(theta0)],
            [0, 1, v0 * delta_t * np.cos(theta0)],
            [0, 0, 1]
        ])

        self.B = np.array([
            [delta_t * np.cos(theta0), 0],
            [delta_t * np.sin(theta0), 0],
            [0, delta_t]
        ])

        # Output matrix
        self.C = np.eye(self.q)  # Assuming full-state observation

    def setCostMatrices(self, Qcoeff, Rcoeff):
        num_states = self.n
        num_controls = self.m

        # Process Qcoeff for state costs
        if np.isscalar(Qcoeff):
            Q = Qcoeff * np.eye(num_states)
        else:
            Q = np.diag(Qcoeff)

        # Process Rcoeff for control costs
        if np.isscalar(Rcoeff):
            R = Rcoeff * np.eye(num_controls)
        else:
            R = np.diag(Rcoeff)

        # Assign matrices to attributes
        self.Q = Q
        self.R = R
