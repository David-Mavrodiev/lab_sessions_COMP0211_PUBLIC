import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")


# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kps, kds, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    kp_vec = kps.copy()
    kd_vec = kds.copy()
    #print(kp_vec)
    #print(kd_vec)
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()

    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []

    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp_vec, kd_vec)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print("current time in seconds",current_time)

    q_mes_all = np.array(q_mes_all)

    if plot:
        # Plot the results for the current joint
        plt.plot(np.arange(len(q_mes_all)) * time_step, q_mes_all[:, joints_id])
        plt.title(f"Joint {joints_id} Position with Kp = {kp_vec[joints_id]} and Kd = {kd_vec[joints_id]}")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint Position [rad]")
        plt.grid(True)
        plt.savefig(f'./week_2/results/joint_{joints_id}', dpi=300)
        plt.show()
    
    return q_mes_all

def perform_frequency_analysis(data, dt, plot=False):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    if plot:
        # Optional: Plot the spectrum
        plt.figure()
        plt.plot(xf, power)
        plt.title("FFT of the signal")
        plt.xlabel("Frequency in Hz")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return xf, power

def find_sustained_oscillations(sim, joint_id, kps, kds, kp_start, kp_step, regulation_displacement, test_duration, tolerance=0.01):
    """
    Search for Ku (ultimate gain) and Tu (oscillation period) by gradually increasing Kp until sustained oscillations are detected.
    """
    cur_kp = kp_start
    kp_vec = kps.copy()
    kd_vec = kds.copy()

    while cur_kp < 1000:
        # updating the kp value for the joint we want to tune
        kp_vec[joint_id] = cur_kp
        kd_vec[joint_id] = 0

        joint_position_data = simulate_with_given_pid_values(sim, kp_vec, kd_vec, joint_id, regulation_displacement, test_duration, plot=False)
        
        dt = sim.GetTimeStep()
        xf, power = perform_frequency_analysis(joint_position_data[:, joint_id], dt, plot=False)

        # Ignore the DC component (0 Hz)
        xf_no_dc = xf[1:]  # Exclude the first element (0 Hz)
        power_no_dc = power[1:]

        print(np.max(power_no_dc))
        # Check for sustained oscillations based on the power spectrum
        if np.max(power_no_dc) > tolerance:
            print(f"Sustained oscillations detected at Kp = {cur_kp}")

            ku = cur_kp  # The ultimate gain where oscillations are first detected

            # Find Tu using the dominant frequency
            dominant_frequency = xf_no_dc[np.argmax(power_no_dc)]
            if dominant_frequency != 0:
                tu = 1 / dominant_frequency
                print(f"Ku = {ku}, Tu = {tu}")
                return ku, tu
            else:
                print(f"Invalid dominant frequency (0 Hz) at Kp = {cur_kp}. Continuing to search...")
        
        cur_kp += kp_step
    
    print("No sustained oscillations detected within the given Kp range.")
    return None, None  # If no oscillations are detected

def compute_performance_metrics(position_data, time_step, tolerance=0.02):    
    # The final stable position is the last value in the position data
    final_position = position_data[-1]
    
    # Overshoot: max deviation from the stable final position
    max_position = np.max(position_data)
    overshoot = (max_position - final_position) / final_position * 100 if final_position != 0 else 0

    # Settling time: time it takes for the position to stay within a tolerance band around the final stable position
    lower_bound = final_position * (1 - tolerance)
    upper_bound = final_position * (1 + tolerance)
    settling_time = None
    
    for i in range(len(position_data)):
        if np.all(position_data[i:] >= lower_bound) and np.all(position_data[i:] <= upper_bound):
            settling_time = i * time_step
            break
    
    if settling_time is None:
        settling_time = len(position_data) * time_step  # If it never settles, use max time

    return settling_time, overshoot

def tune_ziegler_nichols_with_search(sim, joint_id, best_kps, best_kds, kp_start_range, kp_step, regulation_displacement, test_duration):
    """
    Automatically tune Ziegler-Nichols PID parameters by searching for optimal initial gain and step size.
    """
    kps = best_kps.copy() #np.array([0]*dyn_model.getNumberofActuatedJoints()) #best_kps.copy()
    kds = best_kds.copy() #np.array([0]*dyn_model.getNumberofActuatedJoints()) #best_kds.copy()

    best_kp = None
    best_kd = None
    best_performance = float('inf')

    # Loop through a range of initial Kp start values and step sizes
    for kp_start in kp_start_range:
        print(f"Testing Initial Kp = {kp_start}, Step Size = {kp_step}")
        
        # Broad search for Ku and Tu
        ku, tu = find_sustained_oscillations(sim, joint_id, kps, kds, kp_start, kp_step, regulation_displacement, test_duration)
        
        if ku is None or tu is None:
            print("No oscillations detected, moving to the next set of parameters...")
            continue
        
        # Compute Kp and Kd using Ziegler-Nichols rules
        tuned_kp = 0.8 * ku
        tuned_kd = 0.1 * ku * tu
        print(f"Tuned Kp = {tuned_kp}, Tuned Kd = {tuned_kd}")
        
        kps[joint_id] = tuned_kp
        kds[joint_id] = tuned_kd

        # Run the simulation with the tuned parameters and compute performance metrics
        joint_position_data = simulate_with_given_pid_values(sim, kps, kds, joint_id, regulation_displacement, test_duration, plot=False)
        desired_position = init_joint_angles[joint_id] + regulation_displacement
        settling_time, overshoot = compute_performance_metrics(joint_position_data[:, joint_id], desired_position, sim.GetTimeStep())
        
        # Choose a performance score (settling time, overshoot, etc.)
        performance_score = settling_time + 10 * abs(overshoot)

        print(f"Performance Score: {performance_score}")

        # Update best Kp, Kd, and performance score if a better set is found
        if performance_score < best_performance:
            best_performance = performance_score
            best_kp = tuned_kp
            best_kd = tuned_kd

    return best_kp, best_kd, best_performance

def plot_results(regulation_displacement=1, test_duration=20):
    #kps = [36, 8, 36, 8, 20, 8, 16]
    #kds = [22.5, 1.11111111, 15, 3.33333333, 12.5, 1.17647059, 3.63636364]
    #kps = np.array([40, 36, 40, 32, 24, 24, 8])
    #kds = np.array([4, 29, 4, 3, 3, 3, 2])
    #kps = np.array([40, 16, 32, 24, 40, 24, 28])
    #kds = np.array([4, 40, 3, 3, 4, 3, 3])
    #kps = np.array([19, 36, 25, 18, 0, 16, 8])
    #kds = np.array([3, 90, 5, 0, 0, 0, 0])
    #kps = np.array([34, 0, 33, 37, 37, 0, 34])
    #kds = np.array([3.58333333, 2, 3.81818182, 4.27272727, 3.76, 0.66666667, 3.58333333])
    #kps = np.array([19, 27, 20, 36, 21, 21, 23])
    #kds = np.array([3, 22.66666667, 3.05882353, 4, 3.17647059, 3, 3.22222222])
    kps = np.array([36, 8, 36, 8 * 2.4, 20 * 2.5, 8 * 1.5, 16])
    kds = np.array([22.5 * 0.46, 1.11111111 * 0.65, 15 * 0.65, 3.33333333 * 2.2, 12.5, 1.17647059 * 4.7, 3.63636364 * 1.75])

    for joint_id in range(dyn_model.getNumberofActuatedJoints()):
        kp = np.array([0]*dyn_model.getNumberofActuatedJoints())
        kd = np.array([0]*dyn_model.getNumberofActuatedJoints())
        kp[joint_id] = kps[joint_id]
        kd[joint_id] = kds[joint_id]
        simulate_with_given_pid_values(sim, kps, kds, joint_id, regulation_displacement, test_duration, True)

def find_results():
    #joint_id = 0  # Joint ID to tune
    regulation_displacement = 1.0  # Displacement from the initial joint position
    test_duration=20 # in seconds
    gain_step = 1  # Step sizes to test

    #best_kps = np.array([0]*dyn_model.getNumberofActuatedJoints())
    #best_kds = np.array([0]*dyn_model.getNumberofActuatedJoints())

    best_kps = np.array([19, 27, 20, 36, 21, 21, 23])
    best_kds = np.array([3, 22.66666667, 3.05882353, 4, 3.17647059, 3, 3.22222222])

    kp_start_range = np.arange(1000, -1, -1)

    # Tune each joint
    for joint_id in range(dyn_model.getNumberofActuatedJoints()):
        print(f"-----JOINT {joint_id} TUNING-----")
        best_kp, best_kd, best_metrics = tune_ziegler_nichols_with_search(sim, joint_id, best_kps, best_kds, kp_start_range, gain_step, regulation_displacement, test_duration)
        best_kps[joint_id] = best_kp
        best_kds[joint_id] = best_kd

        print(f"Tuned Kp for Joint {joint_id}: {best_kp}")
        print(f"Tuned Kd for Joint {joint_id}: {best_kd}")
        print(f"Performance Metrics for Joint {joint_id}: {best_metrics}")

    print(f"Final Kp values for all joints: {best_kps}")
    print(f"Final Kd values for all joints: {best_kds}")

if __name__ == '__main__':
    plot_results()
    #find_results()
    # Computed PD parameters using the Ziegler-Nichols method
    #Kp's [16.0, 25.6, 16.0, 16.0, 16.0, 16.0, 16.0]
    #Kd's [2.8571428571428568, 1.777777777777778, 2.8571428571428568, 2.8571428571428568, 2.8571428571428568, 2.8571428571428568, 2.6666666666666665]
   