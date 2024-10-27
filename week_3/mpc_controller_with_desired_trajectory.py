import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from regulator_model import RegulatorModel

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints

def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")

def getSystemMatrices(sim, num_joints, damping_coefficients=None):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    time_step = sim.GetTimeStep()
    
    # Initialize the A and B matrices with appropriate sizes
    A = np.zeros((num_states, num_states))  # State transition matrix
    B = np.zeros((num_states, num_controls))  # Control input matrix

    # Filling in the A matrix
    A[:num_joints, num_joints:] = np.eye(num_joints) * time_step
    A[num_joints:, num_joints:] = np.eye(num_joints)
    
    # Filling in the B matrix
    B[num_joints:, :] = np.eye(num_joints) * time_step

    # Optionally add damping to the system
    if damping_coefficients is not None:
        damping = np.diag(damping_coefficients)
        A[num_joints:, num_joints:] -= damping * time_step
    
    return A, B

def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    Q = 10000 * np.eye(num_states)
    #Q[num_joints:, num_joints:] = 0.0  # Penalizing position more than velocity
    
    R = 0.5 * np.eye(num_controls)  # Control input cost matrix
    
    return Q, R

def generate_desired_trajectory(num_joints, steps, init_positions, lower_limits, upper_limits, joint_vel_limits, time_step):
    """
    Generate a desired sinusoidal trajectory for the joints, ensuring that the positions stay within the joint limits.
    """
    t = np.linspace(0, steps * time_step, steps)
    lower_limits = np.array(lower_limits)
    upper_limits = np.array(upper_limits)
    init_positions = np.array(init_positions)
    joint_vel_limits = np.array(joint_vel_limits)

    amplitude = (upper_limits - lower_limits) / 4  # Ensure it stays within limits
    frequency = 0.05  # Frequency in Hz

    q_d_all = []
    qd_d_all = []

    for i in range(steps):
        desired_positions = init_positions + amplitude * np.sin(2 * np.pi * frequency * t[i])
        desired_velocities = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t[i])
        desired_velocities = np.clip(desired_velocities, -joint_vel_limits, joint_vel_limits)  # Clip to joint velocity limits
        q_d_all.append(desired_positions)
        qd_d_all.append(desired_velocities)

    return np.array(q_d_all), np.array(qd_d_all)

def main():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_mes_all, qd_mes_all = [], []
    
    # Define the matrices
    A, B = getSystemMatrices(sim, num_joints)
    Q, R = getCostMatrices(num_joints)
    
    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
    # Compute the matrices needed for MPC optimization
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    
    # Main control loop
    episode_duration = 5
    time_step = sim.GetTimeStep()
    steps = int(episode_duration / time_step)
    sim.ResetPose()
    
    # Generate desired trajectory
    init_positions = sim.GetInitMotorAngles()
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    joint_vel_limits = sim.GetBotJointsVelLimit()
    
    q_d_all, qd_d_all = generate_desired_trajectory(num_joints, steps, init_positions, lower_limits, upper_limits, joint_vel_limits, time_step)
    

    for i in range(steps):
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        
        # Get desired state (position and velocity)
        q_des = q_d_all[i]
        qd_des = qd_d_all[i]
        
        # Calculate position and velocity errors
        position_error = q_mes - q_des
        velocity_error = qd_mes - qd_des
        
        # Update x0_mpc to include the error between current and desired states
        x0_mpc = np.vstack((position_error, velocity_error))
        x0_mpc = x0_mpc.flatten()
        
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[0:num_joints]

        # Control command
        cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        sim.Step(cmd, "torque")  # Simulation step with torque command
        
        # Update the state based on dynamics
        #sim.UpdateMotorAnglesAndVelocities(cmd.tau_cmd)
        
        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
    
    # Plotting the desired vs MPC-tracked trajectories for each joint
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'MPC Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='dashed')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'MPC Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='dashed')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
