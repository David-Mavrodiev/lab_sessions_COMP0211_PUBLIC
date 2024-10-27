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
    
    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)
    
    Returns:
    A: State transition matrix
    B: Control input matrix
    """
    num_states = 2 * num_joints  # Number of states (positions and velocities)
    num_controls = num_joints  # Number of control inputs (accelerations)
    
    time_step = sim.GetTimeStep()  # Time step of the simulation
    
    # Initialize the A and B matrices with appropriate sizes
    A = np.zeros((num_states, num_states))  # State transition matrix
    B = np.zeros((num_states, num_controls))  # Control input matrix

    # Filling in the A matrix
    # Top right block is I * Δt (how positions evolve due to velocities)
    A[:num_joints, num_joints:] = np.eye(num_joints) * time_step
    
    # Bottom right block is I (how velocities evolve with control inputs)
    A[num_joints:, num_joints:] = np.eye(num_joints)
    
    # Filling in the B matrix
    # Control inputs directly affect velocities (accelerations)
    B[num_joints:, :] = np.eye(num_joints) * time_step

    return A, B


def getCostMatrices(num_joints, Q_value):
    """
    Get the cost matrices Q and R for the MPC controller with a variable Q value.
    
    Parameters:
    num_joints: Number of joints
    Q_value: Cost applied to positions (Q matrix scaling factor)
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    Q = Q_value * np.eye(num_states)  # Scale Q by the provided value
    Q[num_joints:, num_joints:] = 0.0  # No penalty for velocity deviations
    
    R = 0.1 * np.eye(num_controls)  # Control input cost matrix
    
    return Q, R


def main():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Q values to test
    Q_values = [1000, 10000, 100000]

    # Storage for position and velocity data for each Q value
    q_mes_all_runs = {q: [] for q in Q_values}
    qd_mes_all_runs = {q: [] for q in Q_values}

    # Horizon length
    N_mpc = 10

    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Run simulation for different Q values
    for Q_value in Q_values:
        # Define system matrices
        A, B = getSystemMatrices(sim, num_joints)
        Q, R = getCostMatrices(num_joints, Q_value)
        
        # Initialize the regulator model
        regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
        # Compute the matrices needed for MPC optimization
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)

        # Reset simulation and collect data for each run
        sim.ResetPose()
        q_mes_all = []
        qd_mes_all = []

        episode_duration = 5
        current_time = 0
        time_step = sim.GetTimeStep()
        steps = int(episode_duration / time_step)

        # Control loop
        for i in range(steps):
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            q_mes_all.append(q_mes)
            qd_mes_all.append(qd_mes)

            # Control command calculations (omitted for brevity)
            x0_mpc = np.hstack((q_mes, qd_mes))
            H_inv = np.linalg.inv(H)
            u_mpc = -H_inv @ F @ x0_mpc
            u_mpc = u_mpc[0:num_joints]
            
            cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
            sim.Step(cmd, "torque")
            
            current_time += time_step

        # Store data for this Q value
        q_mes_all_runs[Q_value] = q_mes_all
        qd_mes_all_runs[Q_value] = qd_mes_all

    # Plotting results for each joint with different Q values
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))

        # Position plot for joint i
        plt.subplot(2, 1, 1)
        for Q_value in Q_values:
            plt.plot([q[i] for q in q_mes_all_runs[Q_value]], label=f'Q = {Q_value}')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        for Q_value in Q_values:
            plt.plot([qd[i] for qd in qd_mes_all_runs[Q_value]], label=f'Q = {Q_value}')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        
        # Save the combined plot for each joint
        plt.savefig(f'./week_3/results2/joint_{i+1}_tracking_Q_comparison.png')
        
        # Optionally display the plot
        plt.show()


if __name__ == '__main__':
    main()
