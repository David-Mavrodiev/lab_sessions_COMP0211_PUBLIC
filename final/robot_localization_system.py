#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        # 注！ 此处更新测量噪声的方差
        # 方差越小，越信任观测值，反之，越相信预测值
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2

# Landmark 地标
# 用于机器人测量自身位置
class Map(object):
    def __init__(self):
        self.landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])

# 自定义的map类 生成更多的landmarks
class Map_self(object):
    def __init__(self):
        grid_spacing = 5
        landmarks = []

        # 30*30 36
        for x in range(-15, 15, grid_spacing):
            for y in range(-15, 15, grid_spacing):
                landmarks.append([x, y])

        # # 50*50 100
        # for x in range(-25, 25, grid_spacing):
        #     for y in range(-25, 25, grid_spacing):
        #         landmarks.append([x, y])
        
        # 将地标列表转换为 NumPy 数组
        self.landmarks = np.array(landmarks)


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

        # 状态更新update
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
            def wrap_angle(angle): return np.arctan2(np.sin(angle), np.cos(angle))
            
            y_pred = []
            C = []
            x_pred = self._x_pred
            
            for lm in self._map.landmarks:
                dx_pred = lm[0] - x_pred[0]
                dy_pred = lm[1] - x_pred[1]
                
                # 计算预测的距离和方位角
                range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
                bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
                bearing_pred = wrap_angle(bearing_pred) # wrap angle!

                # 预测的测量值
                y_pred.extend([range_pred, bearing_pred])

                # 距离的雅可比矩阵
                C_range = np.array([
                    -(dx_pred) / range_pred,
                    -(dy_pred) / range_pred,
                    0
                ])
                
                # 方位角的雅可比矩阵
                C_bearing = np.array([
                    dy_pred / (range_pred ** 2),
                    -dx_pred / (range_pred ** 2),
                    -1
                ])
                C.append(C_range)
                C.append(C_bearing)

            C = np.array(C).reshape(-1, 3)
            y_pred = np.array(y_pred).reshape(-1)
        
            # 计算innovation
            nu = (y_range_bearing - y_pred).reshape(-1)
            for i in range(1, len(nu), 2):
                nu[i] = wrap_angle(nu[i]) # wrap angle!

            # Observation noise cov W
            # range, range, range,..., bearing, bearing, bearing ！！！！！BAD！！！！！
            # num_landmarks = len(self._map.landmarks)
            # # 距离噪声矩阵，(N, N)，
            # W_range_matrix = self._config.W_range * np.eye(num_landmarks)
            # # 方位角噪声矩阵，(N, N)
            # W_bearing_matrix = self._config.W_bearing * np.eye(num_landmarks)
            # W = np.block([
            #     [W_range_matrix, np.zeros((num_landmarks, num_landmarks))],
            #     [np.zeros((num_landmarks, num_landmarks)), W_bearing_matrix]
            # ])
            
            # range, bearing; range, bearing; range, bearing;...
            W = np.diag([self._config.W_range, self._config.W_bearing] * len(self._map.landmarks))
            # print(W.shape)

            # 更新 EKF 状态
            self._do_kf_update(nu, C, W)
            self._x_est[-1] = wrap_angle(self._x_est[-1]) # wrap angle
