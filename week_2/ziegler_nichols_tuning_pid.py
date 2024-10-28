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
    kp_vec = np.array([100]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd_vec = np.array([30]*dyn_model.getNumberofActuatedJoints())
    kd_vec[joints_id] = 0.0
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
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp_vec, kd_vec)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
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

    
    # TODO make the plot for the current joint
    
    
    return q_mes_all
     



def perform_frequency_analysis(data, dt):

    # TODO remove the average from the data signal!

    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf, power)
    plt.title("FFT of the signal")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return xf, power


# TODO Implement the table in thi function




if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune #第一个关节就是id=0
    regulation_displacement = 1.0  # Displacement from the initial joint position
    init_gain=13
    gain_step=1.5 
    max_gain=20 
    test_duration=20 # in seconds
    
    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method

    Ku = None  # 存储最终的Ku值
    Tu = None  # 存储最终的振荡周期
    # Ku_temp = None # 临时存储Ku 用于对比是否持续震荡

    # TODO 之后对于每个关节 添加一个while循环对joint_id进行修改 记得修改Ku Tu的存储方式
    # q_mes_all = simulate_with_given_pid_values(sim, kp_0, joint_id, regulation_displacement)
    for Kp in np.arange(init_gain, max_gain, gain_step):
        print(f"Testing Kp = {Kp}")
        q_mes_all = simulate_with_given_pid_values(sim, Kp, joint_id, regulation_displacement, test_duration)

        # 进行频率分析，查看是否出现持续振荡
        dt = sim.GetTimeStep()
        frequencies, power = perform_frequency_analysis(q_mes_all, dt)

        # TODO 手工检查 如果震荡 则按某个按钮退出当前的循环 并记录当前Kp ku Tu 用于计算 Ku = Kp、Tu = 1 / dominant_frequency
        user_input = input("检测到震荡？按 'y' 记录当前 Kp 为 Ku 并退出循环，按其他键继续测试：")
    
        if user_input.lower() == 'y':
            dominant_frequency_index = np.argmax(power) # !!!!这里获取的时候会出问题！！！获取是为了计算Tu
            dominant_frequency = frequencies[np.argmax(power)] # 获取震荡周期
            # 记录Pu
            Tu = 1 / dominant_frequency  # 计算振荡周期
            # 记录Ku
            Ku = Kp
            break

    # TODO 将当前的Pu和Ku存在数组里
    
    
    if Ku is not None and Tu is not None: # 条件改为Ku和Tu都为7个值
        Kp_final = 0.6 * Ku
        Kd_final = 0.125 * Ku * Tu
        print(f"Final Kp: {Kp_final}, Final Kd: {Kd_final}")

        # 使用最终的PD增益进行仿真
        simulate_with_given_pid_values(sim, Kp_final, joint_id, regulation_displacement, test_duration, plot=True)

   