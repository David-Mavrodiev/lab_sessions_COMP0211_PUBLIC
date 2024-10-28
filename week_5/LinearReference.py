import numpy as np

class LinearReference:
    def __init__(self, q_init=None, max_pos=None):
        """
        Initialize the linear reference.

        Parameters:
        q_init (list or array, optional): Initial positions for each joint. Must be provided.
        """
        # Check if max position is provided
        if max_pos is None:
            raise ValueError("Maximum position 'max_pos' must be provided for the linear reference.")
        else:
            self.max_pos = np.array(max_pos)
        

        # Check if initial position is provided
        if q_init is None:
            raise ValueError("Initial position 'q_init' must be provided for the linear reference.")
        else:
            self.q_init = np.array(q_init)

    def get_values(self, time, coeff=1.0):
        """
        Calculate the position and velocity at a given time for a linear trajectory.

        Parameters:
        time (float or np.array): The time at which to evaluate the position and velocity.
        coeff (float, optional): Coefficient of the linear (affects the "speed" or steepness of the curve).
                                 Defaults to 1.0 for a standard curve.

        Returns:
        tuple: The position (q_d) and velocity (qd_d) at the given time.
        """
        q_d = np.zeros_like(self.q_init)
        qd_d = np.zeros_like(self.q_init)

        # Time threshold before which velocity should start decreasing

        for i in range(len(self.q_init)):
            # normalized_time = np.clip(time, 0, 1)  # Keep time in range [0, 1] for simplicity

            # Calculate position based on normalized time
            q_d[i] = self.q_init[i] + coeff * time
            if q_d[i] > self.max_pos[i]:
                q_d[i] = self.max_pos[i]
                qd_d[i] = 0
            else:
                qd_d[i] = coeff
            
            # Don't use time use max joint position.
        
        return q_d, qd_d