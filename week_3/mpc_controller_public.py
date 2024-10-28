import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from regulator_model import RegulatorModel

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
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
    

def getSystemMatrices(sim, num_joints, damping_coefficients=[0.5, 0.6, 0.2, 0.1, 0.3, 0.35, 0.8]):
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
    num_states = 2 * num_joints # 状态x的大小是关节数的两倍，因为一个状态定义为位置+速度 此处为14
    num_controls = num_joints # 输入u的大小是关节数，因为输入是加速度 此处为7
    
    time_step = sim.GetTimeStep() # 获取时间步长delta t
    
    # TODO: Finish the system matrices

    A = np.eye(num_states) # eye生成对角为1 其余为0的矩阵 注：A为14*14
    # 填充 A 矩阵的位置和速度的时间演化关系
    for i in range(num_joints):
        A[i, num_joints + i] = time_step  # 位置的变化量

    B = np.zeros((num_states, num_controls)) # 生成一个0矩阵（表明加速度对其他部分默认没有影响） 注：B应为7*14
    for i in range(num_joints):
        # B[i, num_joints + i] = time_step
        B[num_joints + i, i] = time_step # 输入对状态的变化量

    # print("matrix A")
    # print(A)
    # print("matrix B")
    # print(B)
    
    ## 注：
    # 图像是否上下镜像无所谓
    # 因为这是一个MPC控制器 该控制器的作用是让机械臂走到某个设定的姿态（此处我们还没有设定这个状态）
    # 所以默认就是回到初始状态 即所有position都是0
    # 因此对于初始位置为0的joint（即奇数的joint），他们的joint本身就达到了目标位置，故图像显示的swing是否镜像其实无所谓
    # 不同方向的swing可能是random的因素造成的，无所谓
    # 主要关注的就是偶数的joint，他们能够回到0位置，就证明sys工作正常

    # 考虑damping之后，系统会存在阻尼，因此需要考虑D矩阵，即为阻尼对系统的影响
    # X_dot = AX + Bu - Dx = (A-D)X + Bu
    D = np.zeros((num_states, num_states))
    if damping_coefficients is not None:
        for i in range(num_joints):
            D[num_joints + i, num_joints + i] = damping_coefficients[i]
        else:
            D = np.zeros((num_states, num_states))
    A = A - D

    return A, B


def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
   
    # Q 矩阵中的元素决定了对状态偏差的敏感度。R矩阵则是对输入控制的敏感度。
    # 矩阵的值越大，MPC 就越关注该部分的误差，希望系统在该状态上接近目标值。
    # 因为我们的目标是让J足够小，因此权重越大，这部分的误差最后越小。
    # Q的作用是对状态的惩罚，这里设置为单位矩阵，即对所有状态的惩罚相同 
    # Q = 10000000 * np.eye(num_states) # 此时系统趋于稳定的速度很快，但是前期会有振荡
    Q = 1000 * np.eye(num_states) # 理论上1000不错，但实际上并没能在模拟步长内回到稳定状态，会用力过猛，导致振荡
    # Q = 100 * np.eye(num_states) # 100是一个比较好的值，可以让系统在一个较短的时间内收敛到稳定状态
    # # Q = 1 * np.eye(num_states) # 没啥用 基本不动
    
    Q[num_joints:, num_joints:] = 0.0 # 该行注释之后，机械臂几乎不动 
    # 因为这是Q[num_joints:, num_joints:]是速度部分惩罚的权重 如果不设置 那么速度就很有可能被惩罚为0 导致不动

    R = 0.1 * np.eye(num_controls)  # Control input cost matrix # np.eye的作用是生成一个对角线为1的矩阵，其余为0

    return Q, R


def getCostMatrices_self(state_weights, control_weights):    
    # Construct Q matrix using state weights
    Q = np.diag(state_weights) # 14*14
    # Construct R matrix using control weights
    R = np.diag(control_weights) # 7*7
    return Q, R

def main():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    state_weights = [1000, 1000, 1000, 1000, 1100, 1361, 1000,
                     0, 0, 0, 0, 0, 0, 0]  # 14
    control_weights = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1]  # 7

    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

    # Define the matrices
    A, B = getSystemMatrices(sim, num_joints)
    Q, R = getCostMatrices(num_joints)
    # Q, R = getCostMatrices_self(state_weights, control_weights) # 自定义的Q和R计算方式
    
    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    # N_mpc = 50 # 等于100的时候卡住了，太大了算不出来
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
    # Compute the matrices needed for MPC optimization
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    
    # Main control loop
    episode_duration = 3 # 模拟的时间长度
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    sim.ResetPose()
    # sim.SetSpecificPose([1, 1, 1, 0.4, 0.5, 0.6, 0.7])
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()

        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        # H_inv = np.linalg.pinv(H) # 伪逆
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_joints]
       
        # Control command
        tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        #print(cmd.tau_cmd) # 输出q qd qdd等信息
        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print(f"Current time: {current_time}")
    
    
    
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
     
    
    
if __name__ == '__main__':
    
    main()