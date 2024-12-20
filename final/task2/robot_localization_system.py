#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2


class Map(object):
    def __init__(self):
        grid_spacing = 5  # Distance between landmarks in meters
        x_min, x_max = -25, 25
        y_min, y_max = -25, 25

        x_coords = np.arange(x_min, x_max + grid_spacing, grid_spacing)
        y_coords = np.arange(y_min, y_max + grid_spacing, grid_spacing)
        self.landmarks = np.array([[x, y] for x in x_coords for y in y_coords])
        # self.landmarks = np.array([
        #     [5, 10],
        #     [15, 5],
        #     [10, 15],

        #     # More landmarks
        #     # [0, 0],
        #     # [20, 0],
        #     # [0, 20],
        #     # [20, 20],
        #     # [10, 0],
        #     # [0, 10],
        #     # [20, 10],
        #     # [10, 20],
        #     # [5, 5],
        #     # [15, 15],
        #     # [5, 15],
        #     # [15, 5]
        # ])


class RobotEstimator(object):

    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map

    # This nethod MUST be called to start the filter
    def start(self):
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    # Predict to the time. The time is fed in to
    # allow for variable prediction intervals.
    def predict_to(self, time):
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # Now predict over a duration dT
        self._predict_over_dt(dt)

    # Return the estimate and its covariance
    def estimate(self):
        return self._x_est, self._Sigma_est

    # This method gets called if there are no observations
    def copy_prediction_to_estimate(self):
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    # This method sets the filter to the initial state
    def _set_estimate_to_initial_conditions(self):
        # Initial estimated state and covariance
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    # Predict to the time
    def _predict_over_dt(self, dt):
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V

        # Predict the new state
        self._x_pred = self._x_est + np.array([
            v_c * np.cos(self._x_est[2]) * dt,
            v_c * np.sin(self._x_est[2]) * dt,
            omega_c * dt
        ])
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]),
                                      np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1,  v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])

        self._kf_predict_covariance(A, self._config.V * dt)

    # Predict the EKF covariance; note the mean is
    # totally model specific, so there's nothing we can
    # clearly separate out.
    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    # Implement the Kalman filter update step.
    def _do_kf_update(self, nu, C, W):

        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred

    def update_from_landmark_range_observations(self, y_range):

        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:

            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are oberving a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))
        
    def update_from_landmark_range_bearing_observations(self, y_range_bearing):
        x_pred = self._x_pred
        num_landmarks = len(self._map.landmarks)
        W_list = []

        y_pred = []
        C = []

        for i, lm in enumerate(self._map.landmarks):
            dx = lm[0] - x_pred[0]
            dy = lm[1] - x_pred[1]
            q = dx**2 + dy**2
            sqrt_q = np.sqrt(q)

            # Predicted measurements
            range_pred = sqrt_q
            bearing_pred = np.arctan2(dy, dx) - x_pred[2]
            bearing_pred = np.arctan2(np.sin(bearing_pred), np.cos(bearing_pred))  # Angle wrapping

            y_pred.extend([range_pred, bearing_pred])

            # Measurement Jacobian
            C_i = np.array([
                [ -dx / sqrt_q, -dy / sqrt_q, 0 ],
                [  dy / q,      -dx / q,     -1 ]
            ])
            C.append(C_i)

            # Measurement noise covariance for this landmark
            W_list.append(self._config.W_range)
            W_list.append(self._config.W_bearing)

        y_pred = np.array(y_pred)
        C = np.vstack(C)
        W = np.diag(W_list)

        # Flatten the measurement array
        y_range_bearing = y_range_bearing.flatten()

        # Innovation
        nu = y_range_bearing - y_pred
        nu[1::2] = np.arctan2(np.sin(nu[1::2]), np.cos(nu[1::2]))  # Angle wrapping for bearings

        self._do_kf_update(nu, C, W)
