import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])


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


def main():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"
    sim, dyn_model, num_joints = init_simulator(conf_file_name)

    # Adjusting floor friction
    floor_friction = 1
    sim.SetFloorFriction(floor_friction)

    # Getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    # Initialize data storage
    base_pos_all, base_bearing_all = [], []

    # Robot parameters
    wheel_radius = 0.11
    wheel_base_width = 0.46

    # Define the matrices
    num_states = 3
    num_controls = 2
    num_outputs = num_states

    # Initialize the regulator model
    N_mpc = 19  # Prediction horizon
    regulator = RegulatorModel(N_mpc, num_outputs, num_controls, num_states)

    # Initial state
    init_pos = np.array([2.0, 3.0])
    init_quat = np.array([0, 0, 0.3827, 0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    x_current = np.array([init_pos[0], init_pos[1], init_base_bearing_])

    # Initial control input
    u_current = np.zeros(num_controls)

    # Define the cost matrices
    Qcoeff = np.array([310, 310, 80.0])
    Rcoeff = 0.5 #np.array([100, 100])
    regulator.setCostMatrices(Qcoeff, Rcoeff)

    # Desired state (goal state)
    x_desired = np.array([0.0, 0.0, 0.0])

    # Control gains for PD controller
    kp = 20.0
    kd = 0.5
    previous_velocity_errors = np.zeros(num_joints)

    # Initialize variables
    u_mpc = np.zeros(num_controls)
    trajectory_nonlinear = []

    num_steps = 2500
    for step in range(num_steps):
        print(f"Step: {step}")

        # Get current state from simulator
        base_pos = sim.GetBasePosition(0)
        base_ori = sim.GetBaseOrientation(0)
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        x_current = np.array([base_pos[0], base_pos[1], base_bearing_])

        # Update system matrices A and B around current state and control
        regulator.updateSystemMatrices(sim, x_current, u_current)

        # Compute the matrices needed for MPC optimization
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)

        # Compute the state error
        x_error = x_current - x_desired
        x_error = x_error.flatten()

        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc_seq = -H_inv @ F @ x_error

        # Extract the first control input
        u_mpc = u_mpc_seq[0:num_controls]

        # Update current control input
        u_current = u_mpc

        # Convert control input (v, omega) to desired wheel velocities
        v_linear = u_mpc[0]
        v_angular = u_mpc[1]

        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(
            v_linear, v_angular, wheel_base_width, wheel_radius)

        desired_wheel_velocities = np.array([
            left_wheel_velocity,
            right_wheel_velocity,
            left_wheel_velocity,
            right_wheel_velocity])

        # Get current wheel velocities
        current_wheel_velocities = sim.GetMotorVelocities(0)

        # Compute velocity errors
        velocity_errors = desired_wheel_velocities - current_wheel_velocities

        # Compute torque commands using PD control
        torque_commands = kp * velocity_errors + kd * (velocity_errors - previous_velocity_errors) / time_step

        previous_velocity_errors = velocity_errors.copy()

        # Apply torque commands to the robot
        cmd = pb.MotorCommands()
        interface_all_wheels = ["torque", "torque", "torque", "torque"]
        cmd.SetControlCmd(torque_commands, interface_all_wheels)

        # Advance the simulation
        sim.Step(cmd, "torque")

        # Store data for plotting
        trajectory_nonlinear.append(x_current)
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)

        # Update current time
        current_time += time_step

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

    # Convert trajectories to arrays
    trajectory_nonlinear = np.array(trajectory_nonlinear)

    # Plot the trajectory
    plt.figure()
    plt.plot(trajectory_nonlinear[:, 0], trajectory_nonlinear[:, 1], label='Robot Trajectory')
    plt.plot(x_desired[0], x_desired[1], 'ro', label='Goal')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('Robot Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()