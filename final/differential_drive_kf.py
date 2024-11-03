import json
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from robot_localization_system import FilterConfiguration, Map, RobotEstimator
from regulator_model import RegulatorModel

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
W_bearing = (np.pi * 0.5 / 180.0) ** 2
# landmarks = []
# for x in range(-25, 25, 5):
#     for y in range(-15, 35, 5):
#         landmarks.append([x, y])
# landmarks = np.array(landmarks)
landmarks = np.array([[5, 10], [15, 5], [10, 15]])


def landmark_range_observations(base_position):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
        range_meas = range_meas + np.random.normal(0, np.sqrt(W))

        y.append(range_meas)

    y = np.array(y)
    return y

def landmark_range_bearing_observations(base_position, base_orientation):
    y = []
    C = []

    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        
        # Range measurement
        range_true = np.sqrt(dx**2 + dy**2)
        range_meas = range_true + np.random.normal(0, np.sqrt(W_range))
        
        # Bearing measurement
        bearing_true = np.arctan2(dy, dx) - base_orientation
        bearing_true = np.arctan2(np.sin(bearing_true), np.cos(bearing_true))  # Angle wrapping
        bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W_bearing))
        
        # Append range and bearing measurements
        y.append([range_meas, bearing_meas])

    # Convert to a numpy array for consistency
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


# def init_simulator(conf_file_name):
#     """Initialize simulation and dynamic model."""
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    
#     ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
#     source_names = ["pybullet"]
    
#     dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
#     num_joints = dyn_model.getNumberofActuatedJoints()
    
#     return sim, dyn_model, num_joints

def init_simulator(conf_file_name, init_position, init_orientation, noise_flag=0):
    """Initialize simulation and dynamic model with a custom initial position and bearing."""
    
    # Load the JSON configuration
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(cur_dir + "/configs/"  + conf_file_name, 'r') as file:
        config = json.load(file)

    print(init_position)
    
    # Update initial position and orientation (bearing in quaternion)
    config["robot_pybullet"]["init_link_base_position"] = [[init_position[0], init_position[1], init_position[2]]]
    config["robot_pybullet"]["init_link_base_orientation"] = [init_orientation]
    config["robot_pybullet"]["noise_flag"] = [noise_flag]

    
    # Write the updated configuration to a temporary file
    temp_conf_file = os.path.join(cur_dir, "temp_conf.json")
    with open(temp_conf_file, 'w') as file:
        json.dump(config, file)
    
    # Initialize the simulator with the updated config file
    sim = pb.SimInterface(temp_conf_file, conf_file_path_ext=cur_dir, use_gui=False)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(temp_conf_file, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def run(init_pos=[2.0, 3.0, 0.0], init_quat=[0,0,0.3827,0.9239], goal_state=None, noise_flag=0):
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    init_pos = init_pos
    init_quat = init_quat
    sim,dyn_model,num_joints=init_simulator(conf_file_name, init_position=init_pos, init_orientation=init_quat, noise_flag=noise_flag)
    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

   
    # Initialize data storage
    base_pos_all, base_bearing_all = [], []
    est_pos_all, est_bearing_all = [], []
    

    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2
   
    
    # Measuring all the state
    
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    # update A,B,C matrices
    # TODO provide state_x_for_linearization,cur_u_for_linearization to linearize the system
    # you can linearize around the final state and control of the robot (everything zero)
    # or you can linearize around the current state and control of the robot
    # in the second case case you need to update the matrices A and B at each time step
    # and recall everytime the method updateSystemMatrices
    init_pos  = np.array(init_pos)
    init_quat = np.array(init_quat)
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])

    # Initialize the Kalman filter
    filter_config = FilterConfiguration([init_pos[0], init_pos[1], init_base_bearing_])
    map_data = Map(landmarks)
    ekf = RobotEstimator(filter_config, map_data)
    ekf.start()
    x_est, Sigma_est = ekf.estimate()

    # Linearize the system around the initial state estimate and control
    cur_state_x_for_linearization = x_est # With Kalman filter
    cur_u_for_linearization = np.zeros(num_controls)

    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    # Define the cost matrices
    Qcoeff = np.array([310, 310, 340.0])
    Rcoeff = [0.5, 0.1]
    regulator.setCostMatrices(Qcoeff,Rcoeff)

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
    
    while current_time < 5:
        # True state propagation (with process noise)
        ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Kalman filter prediction
        ekf.set_control_input(u_mpc)
        ekf.predict_to(current_time + time_step)


        ############################################################################################
    
        # Get the measurements from the simulator ###########################################
         # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        y = landmark_range_bearing_observations(base_pos_no_noise, base_bearing_no_noise_)

    
        # Update the filter with the latest observations
        ekf.update_from_landmark_range_bearing_observations(y)

        ############################################################################################
        
    
        # Get the current state estimate
        x_est, Sigma_est = ekf.estimate()

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
       
   
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        # add this 3 lines if you want to update the A and B matrices at each time step 
        cur_state_x_for_linearization = x_est
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)

        regulator.setCostMatrices(Qcoeff,Rcoeff)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)

        x0_mpc = np.hstack((x_est[:2], x_est[2]))
        x0_mpc = x0_mpc.flatten()

        if goal_state is not None:
            x0_mpc -= goal_state

        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        # u_mpc = -H_inv @ F @ deviation
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls] 
        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)


        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Store data for plotting if necessary
        base_pos_all.append(base_pos_no_noise)
        base_bearing_all.append(base_bearing_no_noise_)
        est_pos_all.append(x_est[:2])
        est_bearing_all.append(x_est[2])

        # Update current time
        current_time += time_step
        print("current time: ", current_time)

    return base_pos_all, base_bearing_all, est_pos_all, est_bearing_all, goal_state


    # Plotting 
    #add visualization of final x, y, trajectory and theta
    # base_pos_all = np.array(base_pos_all)
    # base_bearing_all = np.array(base_bearing_all)
    # plt.figure()
    # plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label='True Path')
    # plt.legend()
    # plt.show()
    # plt.figure()
    # plt.plot(base_bearing_all, label='True Theta')
    # plt.legend()
    # plt.show()
    # # Convert history lists to arrays.
    # base_pos_all = np.array(base_pos_all)
    # base_bearing_all = np.array(base_bearing_all)
    # # Plotting the true path, estimated path, and landmarks.
    # plt.figure()
    # plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label='True Path')
    # plt.scatter(landmarks[:, 0], landmarks[:, 1],
    #             marker='x', color='red', label='Landmarks')
    # plt.legend()
    # plt.show()
    # plt.figure()
    # plt.plot(base_bearing_all, label='True Theta')
    # plt.legend()
    # plt.show()


    
    
    
    

if __name__ == '__main__':
    base_pos_all, base_bearing_all, est_pos_all, est_bearing_all, goal_state = run(noise_flag=1)
    base_pos_all = np.array(base_pos_all)
    est_pos_all = np.array(est_pos_all)

    if goal_state is None:
        goal_state = [0, 0, 0]

    goal_state = np.array(goal_state)
    error = np.linalg.norm(base_pos_all[:, :2] - goal_state[:2], axis=1)

    # Steady-state error (final error value)
    steady_state_error = error[-1]

    print(f"Steady-state error: {steady_state_error}")
    exit()

    plt.figure()
    plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label='True Path', zorder=4)
    plt.plot(est_pos_all[:, 0], est_pos_all[:, 1], label='Estimated Path', linestyle='--', zorder=3)
    plt.scatter(goal_state[0], goal_state[1], color='green', marker='o', label='Goal Point', zorder=5)
    plt.legend()
    plt.grid()
    # plt.savefig("/Users/joefarah/Desktop/Figures/E&C_Final/Task_3/path.png")
    plt.show()

    plt.figure()
    plt.plot(base_bearing_all, label='True Theta')
    plt.plot(est_bearing_all, label='Estimated Theta', linestyle='--')
    plt.legend()
    plt.grid()
    # plt.savefig("/Users/joefarah/Desktop/Figures/E&C_Final/Task_3/bearing.png")
    plt.show()







    