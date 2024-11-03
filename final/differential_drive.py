import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller, regulation_polar_coordinates, regulation_polar_coordinate_quat, wrap_angle, velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])

# grid_spacing = 2  # Distance between landmarks in meters
# x_min, x_max = -5, 5
# y_min, y_max = -5, 5

# x_coords = np.arange(x_min, x_max + grid_spacing, grid_spacing)
# y_coords = np.arange(y_min, y_max + grid_spacing, grid_spacing)
# landmarks = np.array([[x, y] for x in x_coords for y in y_coords])

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
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    return sim, dyn_model, num_joints

def main(N_mpc = 10, max_steps = 5000, Qcoeff = np.array([110, 110, 210.0]), Rcoeff = 2.0, use_terminal_cost = False, use_dynamic_linearization = False):
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim, dyn_model, num_joints = init_simulator(conf_file_name)

    # Adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # Getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    # Initialize data storage
    base_pos_all, base_bearing_all, time_steps, linear_velocities, angular_velocities, x_errors, y_errors, theta_errors, predicted_trajectory = [], [], [], [], [], [], [], [], []

    # Initializing MPC
    num_states = 3
    num_controls = 2

    # Measuring all the state
    C = np.eye(num_states)

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    
    # Linearization choice (fixed or dynamic)
    init_pos = np.array([2.0, 3.0])
    init_quat = np.array([0, 0, 0.3827, 0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.array([0.1, 0.0])

    # Initial linearization around initial state and control
    regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)

    regulator.setCostMatrices(Qcoeff, Rcoeff)
    
    u_mpc = np.array([0, 0])

    ##### Robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    # Define the desired position (goal)
    desired_position = np.array([0, 0])

    step = 0
    while step <= max_steps:
        step = step + 1
        # True state propagation (with process noise)
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Measurements of the current state with noise
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
    
        # MPC Section - Low Level Control
        # Optional dynamic update of A and B matrices if linearizing around the current state each time step
        if use_dynamic_linearization:
            cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
            cur_u_for_linearization = u_mpc
            regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)  # Update A and B at each step
        
        # MPC Optimization Process
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        if use_terminal_cost:
            regulator.computeTerminalWeightMatrix()
            H, F = regulator.compute_H_and_F_with_P(S_bar, T_bar, Q_bar, R_bar)
        else:
            H, F = regulator.compute_H_and_F_without_P(S_bar, T_bar, Q_bar, R_bar)
        #x0_mpc = np.hstack((base_pos[:2], base_bearing_)).flatten()
        # Current robot position and orientation
        x_robot, y_robot = base_pos[0], base_pos[1]
        theta_robot = base_bearing_

        # Position error
        x_error = x_robot - desired_position[0]
        y_error = y_robot - desired_position[1]

        # Orientation error (wrap to [-pi, pi])
        target_angle = np.arctan2(desired_position[1] - y_robot, desired_position[0] - x_robot)

        # Calculate the orientation error as the difference between current orientation and target_angle
        theta_error = wrap_angle(target_angle - theta_robot)

        # Error state for the MPC (includes position and orientation)
        x0_mpc = np.array([x_error, y_error, theta_error])
        
        # Predict the next state based on A, B, and the control input u_mpc
        x0_mpc_pred = x0_mpc.copy()  # Initialize prediction with the current error state
        for _ in range(N_mpc):
            x0_mpc_pred = regulator.A @ x0_mpc_pred + regulator.B @ u_mpc
            predicted_trajectory.append((x0_mpc_pred[0] + desired_position[0], x0_mpc_pred[1] + desired_position[1]))

        # Compute optimal control
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[0:num_controls]
        
        # Prepare control command for the wheels
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        #angular_wheels_velocity_cmd = np.array([left_wheel_velocity, right_wheel_velocity, right_wheel_velocity, left_wheel_velocity])
        
        #angular_wheels_velocity_cmd = np.array([50, -50, 0, 0])
        cmd.SetControlCmd(angular_wheels_velocity_cmd, ["velocity"] * 4)

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Store data for plotting if necessary
        base_pos_all.append(base_pos[:2])  # Only store x, y positions
        base_bearing_all.append(base_bearing_)
        time_steps.append(step)
        linear_velocities.append(u_mpc[0])
        angular_velocities.append(u_mpc[1])
        x_errors.append(x_error)
        y_errors.append(y_error)
        theta_errors.append(theta_error)

        # Update current time
        current_time += time_step

    # Plotting the trajectory and desired position
    # base_pos_all = np.array(base_pos_all)  # Convert to array for easy indexing

    # plt.figure(figsize=(10, 6))
    # plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label='Trajectory', marker='o')
    # plt.plot(desired_position[0], desired_position[1], 'rx', markersize=10, label='Desired Position')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.title('Robot Trajectory and Desired Position')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return {
        'base_pos_all': np.array(base_pos_all),
        'base_bearing_all': np.array(base_bearing_all),
        'desired_position': desired_position,
        'time_steps': np.array(time_steps),
        'linear_velocities': np.array(linear_velocities),
        'angular_velocities': np.array(angular_velocities),
        'x_errors': np.array(x_errors),
        'y_errors': np.array(y_errors),
        'theta_errors': np.array(theta_errors),
        'predicted_trajectory': np.array(predicted_trajectory)
    }

def plot_all_results(multiple_results):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    color_map = cm.get_cmap('tab10', len(multiple_results))  # Color map to get unique colors for each N
    
    for idx, (label, results) in enumerate(multiple_results.items()):
        time = results['time_steps']
        actual_color = color_map(idx)  # Unique color for actual trajectory
        predicted_color = cm.get_cmap('tab20')(idx * 2 + 1)  # Different color for predicted trajectory

        # Trajectory Plot
        axs[0, 0].plot(results['base_pos_all'][:, 0], results['base_pos_all'][:, 1], label=f'{label} Trajectory', color=actual_color)
        if idx == 0:  # Plot desired position only once
            axs[0, 0].plot(results['desired_position'][0], results['desired_position'][1], 'rx', markersize=10, label='Desired Position')
        axs[0, 0].set_xlabel('X Position')
        axs[0, 0].set_ylabel('Y Position')
        axs[0, 0].set_title('Robot Trajectory')
        axs[0, 0].legend()
        axs[0, 0].grid()

        # Predicted vs Actual Trajectory Plot
        axs[0, 1].plot(results['base_pos_all'][:, 0], results['base_pos_all'][:, 1], label=f'{label} Actual Trajectory', marker='o', color=actual_color)
        axs[0, 1].plot(results['predicted_trajectory'][:, 0], results['predicted_trajectory'][:, 1], label=f'{label} Predicted Trajectory', linestyle='--', color=predicted_color)
        if idx == 0:  # Plot desired position only once
            axs[0, 1].plot(results['desired_position'][0], results['desired_position'][1], 'rx', markersize=10, label='Desired Position')
        axs[0, 1].set_xlabel('X Position')
        axs[0, 1].set_ylabel('Y Position')
        axs[0, 1].set_title('Actual vs. Predicted Trajectory')
        axs[0, 1].legend()
        axs[0, 1].grid()

        # Velocity Plot
        axs[1, 0].plot(time, results['linear_velocities'], label=f'{label} Linear Velocity (m/s)', color=actual_color)
        axs[1, 0].plot(time, results['angular_velocities'], label=f'{label} Angular Velocity (rad/s)', linestyle='--', color=predicted_color)
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Velocity')
        axs[1, 0].set_title('Linear and Angular Velocities Over Time')
        axs[1, 0].legend()
        axs[1, 0].grid()

        # Errors Plot
        axs[1, 1].plot(time, results['x_errors'], label=f'{label} X Position Error', color=actual_color)
        axs[1, 1].plot(time, results['y_errors'], label=f'{label} Y Position Error', linestyle='--', color=predicted_color)
        axs[1, 1].plot(time, results['theta_errors'], label=f'{label} Orientation Error', linestyle=':', color=predicted_color)
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Error')
        axs[1, 1].set_title('Position and Orientation Errors Over Time')
        axs[1, 1].legend()
        axs[1, 1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # multiple_results = {}
    # for N in [2, 5, 10, 15, 20]:
    #     results = main(
    #         N_mpc=N,
    #         max_steps=55000,
    #         Qcoeff = np.array([110, 110, 210.0]), #np.array([2, 2, 3]),
    #         Rcoeff = 3.6, #0.5,
    #         use_terminal_cost=False,
    #         use_dynamic_linearization=False
    #     )
    #     multiple_results[f'N={N} (static linearization)'] = results  # Label each result with the corresponding N value

    # # Plot all results in one grid
    # plot_all_results(multiple_results)

    multiple_results = {}
    results_static = main(
        N_mpc=10,
        max_steps=55000,
        Qcoeff = np.array([110, 110, 210.0]),
        Rcoeff = 3.6,
        use_terminal_cost=False,
        use_dynamic_linearization=False
    )
    results_dynamic = main(
        N_mpc=10,
        max_steps=55000,
        Qcoeff = np.array([110, 110, 210.0]),
        Rcoeff = 3.6,
        use_terminal_cost=False,
        use_dynamic_linearization=True
    )
    multiple_results['Static Linearization'] = results_static
    multiple_results['Dynamic Linearization'] = results_dynamic
    plot_all_results(multiple_results)

    # multiple_results = {}
    # N_values = [5, 10, 20]  # Different prediction horizons
    # Q_values = [np.array([100, 100, 200]), np.array([150, 150, 250])]  # Different Q matrices
    # R_values = [2.5, 5.0]  # Different R values
    # for N in N_values:
    #     for Q in Q_values:
    #         for R in R_values:
    #             label = f'N={N}, Q={Q[0]}, R={R}'
    #             multiple_results[label] = main(
    #                 N_mpc = N,
    #                 max_steps = 55000,
    #                 Qcoeff = Q,
    #                 Rcoeff = R,
    #                 use_terminal_cost=False,
    #                 use_dynamic_linearization=True
    #             )
    # plot_all_results(multiple_results)

