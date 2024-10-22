import numpy as np

class PolynomialReference:
    def __init__(self, q_init=None, degree=1):
        """
        Initialize the polynomial reference.

        Parameters:
        degree (int): Degree of the polynomial (1 for linear, 2 for quadratic, 3 for cubic, etc.).
        q_init (list or array, optional): Initial positions for each joint. Must be provided.
        """
        self.degree = degree

        # Check if initial position is provided
        if q_init is None:
            raise ValueError("Initial position 'q_init' must be provided for the polynomial reference.")
        else:
            self.q_init = np.array(q_init)

    def get_values(self, time, coeff=1.0):
        """
        Calculate the position and velocity at a given time for a polynomial trajectory.

        Parameters:
        time (float or np.array): The time at which to evaluate the position and velocity.
        coeff (float, optional): Coefficient of the polynomial (affects the "speed" or steepness of the curve).
                                 Defaults to 1.0 for a standard curve.

        Returns:
        tuple: The position (q_d) and velocity (qd_d) at the given time.
        """
        q_d = np.zeros_like(self.q_init)
        qd_d = np.zeros_like(self.q_init)

        # Calculate position and velocity based on the polynomial degree
        for i in range(len(self.q_init)):
            normalized_time = np.clip(time, 0, 1)  # Keep time in range [0, 1] for simplicity

            if self.degree == 1:  # Linear
                q_d[i] = self.q_init[i] + coeff * normalized_time
                qd_d[i] = coeff
            
            elif self.degree == 2:  # Quadratic
                q_d[i] = self.q_init[i] + coeff * normalized_time ** 2
                qd_d[i] = 2 * coeff * normalized_time
            
            elif self.degree == 3:  # Cubic
                q_d[i] = self.q_init[i] + coeff * (3 * normalized_time ** 2 - 2 * normalized_time ** 3)
                qd_d[i] = coeff * (6 * normalized_time - 6 * normalized_time ** 2)

        return q_d, qd_d