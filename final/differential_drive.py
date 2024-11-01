import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel
from scipy.linalg import solve_discrete_are
from robot_localization_system import FilterConfiguration, Map, Map_self, RobotEstimator

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
# landmarks = np.array([
#             [5, 10],
#             [15, 5],
#             [10, 15]
#         ])
landmarks = Map_self().landmarks # 用更多的landmarks

def landmark_range_observations(base_position):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
       
        y.append(range_meas)

    y = np.array(y)
    return y

# 目前使用的还是3个landmarks 且是在该文件里定义的3个 见上
# 如果要像task1那样用更多的landmarks 需要改动的地方如下：
# 1 前面定义的landmarks 改为 landmarks = Map_self().landmarks
# 2 y_cur_real 要调用 landmark_range_bearing_observations(base_pos, base_bearing)
# 3 map_ 要改为 Map_self()
# 4 ekf_estimator 也要调用含有bearing的那个update
def landmark_range_bearing_observations(base_position, base_bearing):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
        
        # True bearing measurement (with noise)
        bearing_meas = np.arctan2(dy, dx) - base_bearing
        bearing_meas = wrap_angle(bearing_meas)
        
        y.append([range_meas, bearing_meas])

    y = np.array(y).flatten()
    return y

def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def main():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0
   
    # Initialize data storage
    true_pos_all, true_bearing_all = [], []
    estimated_pos_all, estimated_bearing_all = [], []

    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2
    
    # Measuring all the state
    C = np.eye(num_states)
    
    # Horizon length
    # N长度 若太长则加terminal cost P
    N_mpc = 10
    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    
    # 初始化EKF
    filter_conf = FilterConfiguration()
    # map_ = Map()
    map_ = Map_self()  # 使用更多的landmarks
    ekf_estimator = RobotEstimator(filter_conf, map_)
    ekf_estimator.start()   

    # update A,B,C matrices
    # TODO provide state_x_for_linearization,cur_u_for_linearization to linearize the system
    # you can linearize around the final state and control of the robot (everything zero)
    # or you can linearize around the current state and control of the robot
    # in the second case case you need to update the matrices A and B at each time step
    # and recall everytime the method updateSystemMatrices

    # 初始化的位置
    init_pos  = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2]) # rb rf lf lb
    
    # 目标位置 goal position
    # for terminate the simulation only 
    goal_state = np.array([0, 0, 0]) # [x,y,theta]
    position_tolerance = 0.3  # 位置容差
    orientation_tolerance = 0.3  # 角度容差

    # 第一次的时候线性化AB
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    # Define the cost matrices
    # 此处设置Q和R矩阵 Q（状态）越大 动的越猛 R（输入）越大 动的越平滑
    Qcoeff = np.array([310, 320, 310]) # [x, y, theta]
    Rcoeff = 0.5
    regulator.setCostMatrices(Qcoeff,Rcoeff)
   
    # 定义terminal cost P
    Pcoeef = 1
    P = Pcoeef * solve_discrete_are(regulator.A, regulator.B, regulator.Q, regulator.R)
    regulator.setTerminalCost(P)

    u_mpc = np.zeros(num_controls)

    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    v_linear = 0.0
    v_angular = 0.0
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)
    
    while True:
        # True state propagation (with process noise)
        ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Kalman filter prediction
        # 设置 EKF 控制输入
        ekf_estimator.set_control_input(u_mpc)
    
        # Get the measurements from the simulator ###########################################
        # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
        # Measurements of the current state (real measurements with noise) ##################################################################
        # 获取当前真实位置 有观测值噪声W_range
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        # y_cur_real = landmark_range_observations(base_pos)  # 有观测值噪声W_range
        y_cur_real = landmark_range_bearing_observations(base_pos, base_bearing_)  # 有观测值噪声W_range

        # Update the filter with the latest observations
        # 更新 EKF
        ekf_estimator.predict_to(current_time + time_step)
        # ekf_estimator.update_from_landmark_range_observations(y_cur_real)
        ekf_estimator.update_from_landmark_range_bearing_observations(y_cur_real) 

        # Get the current state estimate
        x_est, Sigma_est = ekf_estimator.estimate()
        estimated_pos = x_est[:2] # ekf估计的位置[x,y]
        estimated_bearing = x_est[2] # ekf估计的角度theta

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
   
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        # add this 3 lines if you want to update the A and B matrices at each time step 
        # 此处更新A和B矩阵，线性化当前点，因为实际是非线性的 每次都要更新
        cur_state_x_for_linearization = [estimated_pos[0], estimated_pos[1], estimated_bearing]
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = np.hstack((estimated_pos, estimated_bearing)) # task3更新，使用ekf估计的位置和角度作为状态
        x0_mpc = x0_mpc.flatten()

        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls] 
        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, 
                                                left_wheel_velocity, right_wheel_velocity]) # 原为r l l r 不行的话试试改为r r l l？
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        # 检查当前位置与目标位置的误差
        # pos_error = np.linalg.norm(base_pos[:2] - goal_state[:2])
        # bearing_error = wrap_angle(base_bearing_ - goal_state[2])
        # # 如果误差小于容差，则停止
        # if pos_error < position_tolerance and np.abs(bearing_error) < orientation_tolerance:
        #     print("Goal reached!")
        #     break

        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # Store data for plotting if necessary
        true_pos_all.append(base_pos[:2])
        true_bearing_all.append(base_bearing_)
        estimated_pos_all.append(estimated_pos)  # EKF 估计位置
        estimated_bearing_all.append(estimated_bearing)  # EKF 估计角度

        # Update current time
        current_time += time_step


    # Plotting 
    #add visualization of final x, y, trajectory and theta
    true_pos_all = np.array(true_pos_all)  # [[x1, y1], [x2, y2], ...]
    estimated_pos_all = np.array(estimated_pos_all)  # [theta1, theta2, ...]
    # 创建绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制 x 和 y 的轨迹 task3更新，包括估计和真实位置
    ax1.plot(true_pos_all[:, 0], true_pos_all[:, 1], label='True Trajectory', color='blue')
    ax1.plot(estimated_pos_all[:, 0], estimated_pos_all[:, 1], label='Estimated Trajectory', color='orange', linestyle='--')
    ax1.scatter(true_pos_all[-1, 0], true_pos_all[-1, 1], color='blue', label='Final True Position')
    ax1.scatter(goal_state[0], goal_state[1], color='red', label='Goal Position')
    ax1.legend()
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Robot Trajectory: True vs Estimated')
    ax1.grid()
    ax1.axis('equal')  # 保持x和y轴比例相同

    # 绘制 theta 的变化图 task3更新，包括估计和真实theta
    time_series = np.arange(len(true_bearing_all)) * time_step
    ax2.plot(time_series, true_bearing_all, label='True Theta', color='green')
    ax2.plot(time_series, estimated_bearing_all, label='Estimated Theta', color='purple', linestyle='--')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Theta (rad)')
    ax2.set_title('Theta Change: True vs Estimated')
    ax2.legend()
    ax2.grid()

    # 显示绘图
    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == '__main__':
    main()