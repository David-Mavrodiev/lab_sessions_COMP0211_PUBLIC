import os 
import numpy as np
from numpy.fft import fft, fftfreq
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl


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
    kp_vec = np.array([1000.0]*dyn_model.getNumberofActuatedJoints())
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
        # cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        # regressor_all = np.vstack((regressor_all, cur_regressor))

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print("current time in seconds",current_time)

    
    # TODO make the plot for the current joint
    if plot:
        x = np.array(q_mes_all)[:, joints_id:joints_id+1]
        plt.figure()
        plt.plot(x)
        plt.axhline(y=np.max(x), linestyle='--', color='r')
        plt.axhline(y=np.min(x), linestyle='--', color='r')
        plt.title(f"Joint {joints_id} position tracking with kp: {kp}")
        plt.xlabel("Time in seconds")
        plt.ylabel("Joint angle in rad")
        plt.legend(["q_mes", "q_max", "q_min"])
        plt.grid(True)
        plt.show()
    
    return q_mes_all
     


def perform_frequency_analysis(data, dt, kp, plot=False):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    if plot:
        plt.figure()
        plt.plot(xf, power)
        plt.title(f"FFT of the signal with kp: {kp}")
        plt.xlabel("Frequency in Hz")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return xf, power


# TODO Implement the table in this function
def show_table(kus, tus):
    tables = open("tables.txt", "w")
    for joint_id in range(num_joints):
        Ku = kus[joint_id]
        Tu = tus[joint_id]
        data = {
            "Control Type": ["P", "PI", "PD", "classic PID", "Pessen Integral Rule", "some overshoot", "no overshoot"],
            "Kp": [f"{0.5*Ku:.3f}", f"{0.45*Ku:.3f}", f"{0.8*Ku:.3f}", f"{0.6*Ku:.3f}", f"{0.7*Ku:.3f}", f"{0.33*Ku:.3f}", f"{0.20*Ku:.3f}"],
            "Ti": ["–", f"{0.83*Tu:.3f}", "–", f"{0.5*Tu:.3f}", f"{0.4*Tu:.3f}", f"{0.50*Tu:.3f}", f"{0.50*Tu:.3f}"],
            "Td": ["–", "–", f"{0.125*Tu:.3f}", f"{0.125*Tu:.3f}", f"{0.15*Tu:.3f}", f"{0.33*Tu:.3f}", f"{0.33*Tu:.3f}"],
            "Ki": ["–", f"{0.54*Ku/Tu:.3f}", "–", f"{1.2*Ku/Tu:.3f}", f"{1.75*Ku/Tu:.3f}", f"{0.66*Ku/Tu:.3f}", f"{0.40*Ku/Tu:.3f}"],
            "Kd": ["–", "–", f"{0.10*Ku*Tu:.3f}", f"{0.075*Ku*Tu:.3f}", f"{0.105*Ku*Tu:.3f}", f"{0.11*Ku*Tu:.3f}", f"{0.066*Ku*Tu:.3f}"]
        }

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Use tabulate to print the DataFrame as a table
        tables.write(f"Ziegler-Nichols Method Table for Joint {joint_id+1}\n")
        tables.write(tabulate(df, headers='keys', tablefmt='grid', showindex=False, colalign=("center", "center", "center", "center", "center", "center")))
        tables.write("\n\n")


if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune
    regulation_displacement = 1.0  # Displacement from the initial joint position
    init_gain=18.0  # Initial Kp value
    gain_step=1.0
    max_gain=21 
    test_duration=20 # in seconds
    
    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
    kps = [init_gain for i in range(7)] # initial kp values
    kds = [0,0,0,0,0,0,0] # initial kd values
    kus = [0,0,0,0,0,0,0] # ultimate gains
    tus = [0,0,0,0,0,0,0] # ultimate periods

    for joint_id in range(num_joints):
        cur_kp = kps[joint_id]

        while cur_kp < max_gain:
            print(f"Testing Kp: {0.8 * cur_kp:.1f} for Joint {joint_id+1}")
            q_mes_all = simulate_with_given_pid_values(sim, cur_kp, joint_id, regulation_displacement, test_duration)

            q_mes_joint = np.array(q_mes_all)[:, joint_id]

            dt = sim.GetTimeStep()

            xf, power = perform_frequency_analysis(q_mes_joint, dt, cur_kp)
        
            xf_no_dc = xf[1:] # Remove DC component
            power_no_dc = power[1:] # Remove DC component

            max_power = np.max(power_no_dc)

            if max_power > 0.1:
                print(f"max power: {max_power}")
                print(f"Oscillation detected with Kp = {0.8 * cur_kp:.1f}.")
                ku = cur_kp
                
                dominant_freq = xf_no_dc[np.argmax(power_no_dc)]
                print(f"Dominant frequency = {dominant_freq} Hz")
                if dominant_freq > 0.1:
                    tu = 1 / dominant_freq
                    kps[joint_id] = 0.8 * ku
                    kds[joint_id] = 0.1 * ku * tu
                    kus[joint_id] = ku
                    tus[joint_id] = tu
                # break

            cur_kp += gain_step
    
    print(f"Final Kp values: {kps}")
    show_table(kus, tus)

   