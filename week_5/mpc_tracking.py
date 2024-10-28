import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from tracker_model import TrackerModel
from LinearReference import LinearReference

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
    num_states = 2 * num_joints
    num_controls = num_joints
    
    time_step = sim.GetTimeStep()
    
    # Initialize A matrix
    A = np.eye(num_states)
    
    # Upper right quadrant of A (position affected by velocity)
    A[:num_joints, num_joints:] = np.eye(num_joints) * time_step
    
    # Lower right quadrant of A (velocity affected by damping)
    if damping_coefficients is not None:
        damping_matrix = np.diag(damping_coefficients)
        A[num_joints:, num_joints:] = np.eye(num_joints) - time_step * damping_matrix
    
    # Initialize B matrix
    B = np.zeros((num_states, num_controls))
    
    # Lower half of B (control input affects velocity)
    B[num_joints:, :] = np.eye(num_controls) * time_step
    
    return A, B

# Example usage:
# sim = YourSimulationObject()
# num_joints = 6  # Example: 6-DOF robot
# damping_coefficients = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]  # Example damping coefficients
# A, B = getSystemMatrices(sim, num_joints, damping_coefficients)


def getCostMatrices(num_joints, p_w=10000, v_w=10):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    # Q = 1 * np.eye(num_states)  # State cost matrix
    # p_w = 1000
    # v_w = 1
    Q_diag = np.array([p_w, p_w, p_w,p_w, p_w, p_w,p_w, v_w, v_w, v_w,v_w, v_w, v_w,v_w])
    Q = np.diag(Q_diag)
    # Q[num_joints:, num_joints:] = 0.0
    
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
    regressor_all = np.array([])

    # Define the matrices
    A, B = getSystemMatrices(sim, num_joints)

    p_ws = [10000, 10000, 100000] # To track position only, set p_w to 10000000
    v_ws = [100, 10, 100] # To track position only, set v_w to 0
    colors = ['C0', 'g', 'b']
    results = {}

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    # ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    max_pos = sim.GetBotJointsLimit()[1] # Maximum joint position limits
    ref = LinearReference(sim.GetInitMotorAngles(), max_pos)  # Initialize the linear reference

    # f = open(f"week_5/results.txt", "a")

    for p_w, v_w in zip(p_ws, v_ws):
        q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

        Q, R = getCostMatrices(num_joints, p_w, v_w)
        # Q, R = getCostMatrices(num_joints) # Default values
    
        # Measuring all the state
        num_states = 2 * num_joints
        C = np.eye(num_states)
        
        # Horizon length
        N_mpc = 10

        # Initialize the regulator model
        tracker = TrackerModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
        # Compute the matrices needed for MPC optimization
        S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar = tracker.propagation_model_tracker_fixed_std()
        H,Ftra = tracker.tracker_std(S_bar, T_bar, Q_hat, Q_bar, R_bar)
        

        # Main control loop
        episode_duration = 5 # duration in seconds
        current_time = 0
        start_time = time.time()
        time_step = sim.GetTimeStep()
        steps = int(episode_duration/time_step)
        sim.ResetPose()
        # sim.SetSpecificPose([1, 1, 1, 0.4, 0.5, 0.6, 0.7])
        # testing loop
        u_mpc = np.zeros(num_joints)
        for i in range(steps):
            # measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            # Compute sinusoidal reference trajectory
            # Ensure q_init is within the range of the amplitude
            
            x0_mpc = np.vstack((q_mes, qd_mes))
            x0_mpc = x0_mpc.flatten()
            x_ref = []
            # generate the predictive trajectory for N steps
            for j in range(N_mpc):
                q_d, qd_d = ref.get_values(current_time + j*time_step)
                # here i need to stack the q_d and qd_d
                x_ref.append(np.vstack((q_d.reshape(-1, 1), qd_d.reshape(-1, 1))))
            
            x_ref = np.vstack(x_ref).flatten()

            # Compute the optimal control sequence
            u_star = tracker.computesolution(x_ref, x0_mpc, u_mpc, H, Ftra)
            # Return the optimal control sequence
            u_mpc += u_star[:num_joints]
            
            # Control command
            tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)  # Zero torque command
            cmd.SetControlCmd(tau_cmd, ["torque"]*7) 
            sim.Step(cmd, "torque")  # Simulation step with torque command

            # print(cmd.tau_cmd)
            # Exit logic with 'q' key
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')
            if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                break
            
            #simulation_time = sim.GetTimeSinceReset()

            # Store data for plotting
            q_mes_all.append(q_mes)
            qd_mes_all.append(qd_mes)

            q_d, qd_d = ref.get_values(current_time)

            q_d_all.append(q_d)
            qd_d_all.append(qd_d)

            # time.sleep(0.01)  # Slow down the loop for better visualization
            # get real time
            current_time += time_step
            print(f"Time: {current_time}")

        results[(p_w, v_w)] = (q_mes_all, qd_mes_all)
        break
        # f.write(f"Elapsed time for P_w = {p_w}, V_w = {v_w}: {time.time() - start_time:.2f} seconds")
        
        
    
    
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        for idx, (p_w, v_w) in enumerate(results.keys()):
            plt.plot([q[i] for q in results[(p_w, v_w)][0]], label=rf'Measured Position - $P_w = {p_w}$', color=colors[idx])
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position', linestyle='--', color='orange')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        for idx, (p_w, v_w) in enumerate(results.keys()):
            plt.plot([qd[i] for qd in results[(p_w, v_w)][1]], label=rf'Measured Velocity - $V_w = {v_w}$', color=colors[idx])
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity', linestyle='--', color='orange')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'/Users/joefarah/Desktop/Figures/E&C_Lab3/joint_{i+1}_tracking_linear.png', dpi=300)
        # plt.show()

    # fig, axes = plt.subplots(num_joints, 2, figsize=(14, num_joints * 3))  

    # for i in range(num_joints):
    #     # Position plot for joint i
    #     ax_pos = axes[i, 0]
    #     for idx, (p_w, v_w) in enumerate(results.keys()):
    #         ax_pos.plot([q[i] for q in results[(p_w, v_w)][0]], label=rf'Measured Position - $P_w = {p_w}$', color=colors[idx])
    #     ax_pos.plot([q[i] for q in q_d_all], label='Desired Position', linestyle='--', color='orange')
    #     ax_pos.set_title(f'Joint {i+1}', fontsize=10)
    #     ax_pos.set_xlabel('Time steps', fontsize=8)
    #     ax_pos.set_ylabel('Position', fontsize=8)
    #     ax_pos.legend(fontsize=6, loc='upper right')

    #     # Velocity plot for joint i
    #     ax_vel = axes[i, 1]
    #     for idx, (p_w, v_w) in enumerate(results.keys()):
    #         ax_vel.plot([qd[i] for qd in results[(p_w, v_w)][1]], label=f'Measured Velocity - V_w = {v_w}', color=colors[idx])
    #     ax_vel.plot([qd[i] for qd in qd_d_all], label='Desired Velocity', linestyle='--', color='orange')
    #     ax_vel.set_title(f'Joint {i+1}', fontsize=10)
    #     ax_vel.set_xlabel('Time steps', fontsize=8)
    #     ax_vel.set_ylabel('Velocity', fontsize=8)
    #     ax_vel.legend(fontsize=6, loc='upper right')

    # # Adjust layout
    # plt.tight_layout(pad=2.0)
    # # plt.savefig('/Users/joefarah/Desktop/Figures/E&C_Lab3/tracking.png', dpi=300)  # Specify the path to save the figure
    # plt.show()
     
    
    
if __name__ == '__main__':
    
    main()