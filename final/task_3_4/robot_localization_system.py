#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class FilterConfiguration(object):
    def __init__(self, x0=np.array([0, 0, np.pi / 4]), Sigma0=np.diag([1.0, 1.0, 0.5]) ** 2):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = x0
        self.Sigma0 = Sigma0


class Map(object):
    def __init__(self, landmarks=None):
        # self.landmarks = np.array([
        #     [5, 10],
        #     [15, 5],
        #     [10, 15]
        # ])

        
        if landmarks is not None:
            self.landmarks = np.array(landmarks)
        else:
            """ Task 1 """
            # # Activity 2
            landmarks = []
            for x in range(-25, 25, 5):
                for y in range(-15, 35, 5):
                    landmarks.append([x, y])
            self.landmarks = np.array(landmarks)

            # Activity 4
            # landmarks = []
            # for x in range(-15, 15, 5):
            #     for y in range(-15, 15, 5):
            #         landmarks.append([x, y])
            # self.landmarks = np.array(landmarks)




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
        # Predicted landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred

        for lm in self._map.landmarks:  
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]

            # Predicted range and bearing
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
            bearing_pred = np.arctan2(np.sin(bearing_pred), np.cos(bearing_pred))  # Angle wrapping

            # Append predicted range and bearing
            y_pred.extend([range_pred, bearing_pred])

            # Jacobian of the measurement model for range
            C_range = np.array([
                -dx_pred / range_pred,
                -dy_pred / range_pred,
                0
            ])

            # Jacobian of the measurement model for bearing
            C_bearing = np.array([
                dy_pred / (range_pred**2),
                -dx_pred / (range_pred**2),
                -1
            ])

            # Stack the range and bearing Jacobians vertically for this landmark
            C.append(C_range)
            C.append(C_bearing)

        # Convert lists to arrays
        C = np.vstack(C)  # Stack all rows vertically to get a (2N x 3) matrix
        y_pred = np.array(y_pred)

        # Innovation: actual measurements (y_range_bearing) - predicted (y_pred)
        nu = y_range_bearing - y_pred

        # Wrap the bearing differences in the innovation to [-π, π]
        for i in range(len(self._map.landmarks)):
            nu[2 * i + 1] = np.arctan2(np.sin(nu[2 * i + 1]), np.cos(nu[2 * i + 1]))

        # Observation noise covariance for both range and bearing
        W_landmarks = np.diag([self._config.W_range, self._config.W_bearing] * len(self._map.landmarks))

        # Kalman filter update
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap the final orientation estimate
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]), np.cos(self._x_est[-1]))