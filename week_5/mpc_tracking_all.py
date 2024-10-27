import numpy as np
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from tracker_model import TrackerModel
from custom_reference import CustomReference

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=True)
    
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
    

def getSystemMatricesContinuos(num_joints, damping_coefficients=None):
    """Get the system matrices A and B according to the dimensions of the state and control input."""
    num_states = 2 * num_joints
    num_controls = num_joints
    
    # Initialize A matrix
    A = np.zeros((num_states, num_states))
    
    # Upper right quadrant of A (position affected by velocity)
    A[:num_joints, num_joints:] = np.eye(num_joints) 
    
    # Initialize B matrix
    B = np.zeros((num_states, num_controls))
    
    # Lower half of B (control input affects velocity)
    B[num_joints:, :] = np.eye(num_controls) 
    
    return A, B


def getCostMatrices(num_joints):
    """Get the cost matrices Q and R for the MPC controller."""
    num_states = 2 * num_joints
    num_controls = num_joints
    
    p_w = 10000
    v_w = 10000
    Q_diag = np.array([p_w, p_w, p_w, p_w, p_w, p_w, p_w, v_w, v_w, v_w, v_w, v_w, v_w, v_w])
    Q = np.diag(Q_diag)
    
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
    
    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

    # Define the matrices
    A, B = getSystemMatricesContinuos(num_joints)
    Q, R = getCostMatrices(num_joints)
    
    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    tracker = TrackerModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states, sim.GetTimeStep())
    # Compute the matrices needed for MPC optimization
    S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar = tracker.propagation_model_tracker_fixed_std()
    H,Ftra = tracker.tracker_std(S_bar, T_bar, Q_hat, Q_bar, R_bar)
    
    # Sinusoidal reference for both position and velocity
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Amplitudes for joints
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Frequencies for joints

    #amplitude = np.array(amplitudes)
    #frequency = np.array(frequencies)
    #ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize reference
    # Choose reference type
    trajectory_type = "linear"  # Choose 'linear' or 'polynomial'
    if trajectory_type == "linear":
        #Linear reference (constant velocity)
        ref = CustomReference(trajectory_type="linear", initial_position=sim.GetInitMotorAngles())
    else:
        #Polynomial reference (e.g., q(t) = a0 + a1*t + a2*t^2 for each joint)
        coefficients = [0.5, 0.2, 0.05]  # Example coefficients
        ref = CustomReference(trajectory_type="polynomial", coefficients=coefficients)

    # Main control loop
    episode_duration = 5  # Duration in seconds
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration / time_step)
    sim.ResetPose()
    
    u_mpc = np.zeros(num_joints)
    for i in range(steps):
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)

        x0_mpc = np.vstack((q_mes, qd_mes)).flatten()
        x_ref = []

        # Generate predictive trajectory for N steps (both position and velocity)
        for j in range(N_mpc):
            q_d, qd_d = ref.get_values(current_time + j * time_step)
            x_ref.append(np.vstack((q_d.reshape(-1, 1), qd_d.reshape(-1, 1))))
        
        x_ref = np.vstack(x_ref).flatten()

        # Compute the optimal control sequence
        u_star = tracker.computesolution(x_ref, x0_mpc, u_mpc, H, Ftra)
        u_mpc += u_star[:num_joints]
       
        # Apply control command
        cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        sim.Step(cmd, "torque")

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_d)
        qd_d_all.append(qd_d)

        current_time += time_step

    # Plotting position and velocity tracking
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
