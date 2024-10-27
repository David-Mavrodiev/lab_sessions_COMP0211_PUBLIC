import numpy as np

class CustomReference:
    def __init__(self, trajectory_type="linear", coefficients=None, initial_position=None):
        """
        Initializes a custom reference generator.

        Parameters:
        - trajectory_type: Type of reference ('linear' or 'polynomial')
        - coefficients: Coefficients for the polynomial (if polynomial trajectory)
        - initial_position: Initial joint positions (if linear trajectory)
        """
        self.trajectory_type = trajectory_type
        self.coefficients = coefficients if coefficients is not None else []
        self.initial_position = initial_position if initial_position is not None else np.zeros(7)  # Assuming 7 joints

    def get_values(self, current_time):
        """
        Returns the reference position and velocity at the current time.

        Parameters:
        - current_time: The current time step

        Returns:
        - q_d: Desired position (array for each joint)
        - qd_d: Desired velocity (array for each joint)
        """
        if self.trajectory_type == "linear":
            return self._get_linear_reference(current_time)
        elif self.trajectory_type == "polynomial":
            return self._get_polynomial_reference(current_time)
        else:
            raise ValueError("Invalid trajectory type. Use 'linear' or 'polynomial'.")

    def _get_linear_reference(self, current_time):
        """Generates a linear trajectory: q(t) = initial_position + velocity * t"""
        velocity = np.ones(7)  # Define constant velocity for each joint
        q_d = self.initial_position + velocity * current_time
        qd_d = velocity
        return q_d, qd_d

    def _get_polynomial_reference(self, current_time):
        """Generates a polynomial trajectory: q(t) = a0 + a1*t + a2*t^2 + ..."""
        q_d = np.zeros(7)  # Initialize desired positions
        qd_d = np.zeros(7)  # Initialize desired velocities
        for i in range(len(self.coefficients)):
            q_d += self.coefficients[i] * (current_time ** i)
            if i > 0:
                qd_d += i * self.coefficients[i] * (current_time ** (i - 1))
        return q_d, qd_d
