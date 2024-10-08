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
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd = np.array([0]*dyn_model.getNumberofActuatedJoints())
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
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
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
        plt.title(f"Joint {joints_id} Position with Kp = {kp}")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint Position [rad]")
        plt.grid(True)
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

if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune
    regulation_displacement = 1.0  # Displacement from the initial joint position
    init_gain=15
    gain_step=0.1
    max_gain=10000 
    test_duration=20 # in seconds

    kps = np.array([0]*dyn_model.getNumberofActuatedJoints())
    kds = np.array([0]*dyn_model.getNumberofActuatedJoints())

    #simulate_with_given_pid_values(sim, 16.0, 6, regulation_displacement, test_duration, True)

    while joint_id < dyn_model.getNumberofActuatedJoints():
        print(f"-----JOINT {joint_id} SIMULATION-----")
        cur_kp = init_gain
        while cur_kp < max_gain:
            joint_position_data = simulate_with_given_pid_values(sim, cur_kp, joint_id, regulation_displacement, test_duration, False)
            print(joint_position_data.shape)

            dt = sim.GetTimeStep()
            xf, power = perform_frequency_analysis(joint_position_data[:, joint_id], dt, False)
            
            # Ignore the DC component (0 Hz)
            xf_no_dc = xf[1:]  # Exclude the first element (0 Hz)
            power_no_dc = power[1:]

            print(f"Xf (no DC) = {xf_no_dc}, Power (no DC) = {power_no_dc}")

            # Check for sustained oscillations (based on power spectrum analysis)
            if np.max(power_no_dc) > 0.1:  # Adjust this threshold if necessary
                print(f"Sustained oscillations detected at Kp = {cur_kp}")
                
                ku = cur_kp
                
                # Find Tu using the dominant frequency
                dominant_frequency = xf_no_dc[np.argmax(power_no_dc)]  # Find dominant frequency excluding 0 Hz
                if dominant_frequency != 0:
                    tu = 1 / dominant_frequency
                    kps[joint_id] = 0.8 * ku
                    kds[joint_id] = 0.1 * ku * tu
                    print(f"Ku = {ku}, Tu = {tu}")
                    print(f"Tuned Proportional Gain (Kp): {kps[joint_id]}")
                    print(f"Tuned Derivative Gain (Kd): {kds[joint_id]}")

                    break
                else:
                    print(f"Invalid dominant frequency (0 Hz) at Kp = {cur_kp}. Continuing to search...")

            cur_kp += gain_step
        joint_id += 1

    print(f"Kp's = {kps}")
    print(f"Kd's = {kds}")

    # Computed PD parameters using the Ziegler-Nichols method
    #Kp's [16.0, 25.6, 16.0, 16.0, 16.0, 16.0, 16.0]
    #Kd's [2.8571428571428568, 1.777777777777778, 2.8571428571428568, 2.8571428571428568, 2.8571428571428568, 2.8571428571428568, 2.6666666666666665]
   