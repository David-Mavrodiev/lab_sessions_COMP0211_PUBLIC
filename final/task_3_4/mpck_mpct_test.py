import numpy as np
from differential_drive import run as run_gt
from differential_drive_kf import run as run_kf
import matplotlib.pyplot as plt

class Test:
    def __init__(self, init_pos, goal_state, init_quat):
        self.init_pos = init_pos
        self.init_quat = init_quat
        self.goal_state = goal_state

    def test_gt(self, noise_flag=0): 
        print(f"Running MPCT test from {self.init_pos} to {self.goal_state}")
        return run_gt(init_pos=self.init_pos, init_quat=self.init_quat, goal_state=self.goal_state, noise_flag=noise_flag)

    def test_kf(self, noise_flag=0):
        print(f"Running MPCK test from {self.init_pos} to {self.goal_state}")
        return run_kf(init_pos=self.init_pos, init_quat=self.init_quat, goal_state=self.goal_state, noise_flag=noise_flag)
    
    def get_init_pos(self):
        return self.init_pos
    
    def get_goal_state(self):
        return self.goal_state
    
def calculate_metrics(pos, goal, time):
    pos = np.array(pos)
    goal = np.array(goal)

    # Compute error over time (Euclidean distance)
    error = np.linalg.norm(pos[:, :2] - goal[:2], axis=1)

    # Steady-state error (final error value)
    steady_state_error = error[-1]

    # Settling time threshold 
    threshold = 0.5  # 50 cm

    # Settling time calculation
    settling_time = time[-1]  # Default to the last time if settling time not found
    for idx in range(len(error)):
        if error[idx] <= threshold and np.all(error[idx:] <= threshold):
            settling_time = time[idx]
            break

    # Overshoot and undershoot calculation along the movement direction
    initial_pos = pos[0][:2]
    goal_pos = goal[:2]

    movement_vector = goal_pos - initial_pos
    movement_distance = np.linalg.norm(movement_vector)

    if movement_distance == 0:
        # No movement; overshoot and undershoot are zero
        overshoot_percentage = 0.0
        undershoot_percentage = 0.0
    else:
        movement_direction = movement_vector / movement_distance

        # Compute scalar projection of positions onto the movement vector
        positions_relative_to_initial = pos[:, :2] - initial_pos
        scalar_projections = np.dot(positions_relative_to_initial, movement_direction)

        # Expected final projection (should be equal to movement_distance)
        expected_projection = movement_distance

        # Maximum projection along movement direction
        max_projection = np.max(scalar_projections)

        # Overshoot occurs if max_projection > expected_projection
        if max_projection > expected_projection:
            overshoot_amount = max_projection - expected_projection
            overshoot_percentage = (overshoot_amount / movement_distance) * 100
            undershoot_percentage = 0.0
        else:
            # Undershoot is the remaining distance to the goal
            undershoot_amount = expected_projection - max_projection
            undershoot_percentage = (undershoot_amount / movement_distance) * 100
            overshoot_percentage = 0.0

    return steady_state_error, settling_time, overshoot_percentage, undershoot_percentage
    

if __name__ == "__main__":
    # Define test scenarios
    tests = {
        "Straight-Line Test": Test(
            init_pos=[0, 0, 0], 
            goal_state=[3, 0, 0],
            init_quat=[0, 0, 0, 1.0]  # No rotation along z-axis for straight movement along x
        ),

        "Diagonal Movement Test": Test(
            init_pos=[0, 0, 0],
            goal_state=[3, 3, 0],
            init_quat=[0, 0, 0.3827, 0.9239]
        ),

        "Sharp Turn Test": Test(
            init_pos=[1, 1, 0], 
            goal_state=[1, 3, 0],
            init_quat=[0, 0, 0.7071, 0.7071]  # 90-degree rotation for a sharp turn
        ),

        "Reverse Movement Test": Test(
            init_pos=[2, 2, 0], 
            goal_state=[0, 2, 0],
            init_quat=[0, 0, 1.0, 0]  # 180-degree rotation for backward movement
        ),
    }

    pos_all_gt, bearing_all_gt = [], []
    pos_all_kf, bearing_all_kf = [], []
    starts, goals = [], []
    results = {"MPCT": [], "MPCK": []}
    time  = np.arange(0.001, 5.001, 0.001)

    # Run each test for MPCT and MPCK with noise
    for name, test in tests.items():
        print(f"\n{name}")

        starts.append(test.get_init_pos())
        goals.append(test.get_goal_state())

        # Ground truth (GT) without noise
        pos_gt, bearing_gt, _ = test.test_gt()
        pos_all_gt.append(pos_gt)
        bearing_all_gt.append(bearing_gt)
        results["MPCT"].append(calculate_metrics(pos_gt, test.get_goal_state(), time))
        
        # Kalman Filter (KF) with noise
        pos_kf, bearing_kf, _, _, _ = test.test_kf(noise_flag=1)
        pos_all_kf.append(pos_kf)
        bearing_all_kf.append(bearing_kf)
        results["MPCK"].append(calculate_metrics(pos_kf, test.get_goal_state(), time))

    # Arrange the subplots in a 2-column grid
    num_tests = len(tests)
    num_rows = (num_tests + 1) // 2  # Calculate the number of rows needed for 2 columns

    pos_all_gt = np.array(pos_all_gt)
    pos_all_kf = np.array(pos_all_kf)
    goals = np.array(goals)
    starts = np.array(starts)

    for i, (name, pos_gt, pos_kf) in enumerate(zip(tests.keys(), pos_all_gt, pos_all_kf)):
        # Calculate Euclidean distance error for MPCT and MPCK
        goal_pos = goals[i][:2]
        error_gt = np.linalg.norm(pos_gt[:, :2] - goal_pos, axis=1)
        error_kf = np.linalg.norm(pos_kf[:, :2] - goal_pos, axis=1)

        # Create a new figure for each test case
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(f"{name} - Trajectory and Error")

        # Trajectory subplot
        ax_traj = fig.add_subplot(1, 2, 1)
        ax_traj.scatter(starts[i, 0], starts[i, 1], color='green', marker='o', label='Start', zorder=4)
        ax_traj.scatter(goals[i, 0], goals[i, 1], color='red', marker='x', label='Goal', zorder=4)
        ax_traj.plot(pos_gt[:, 0], pos_gt[:, 1], label="GT", zorder=2)
        ax_traj.plot(pos_kf[:, 0], pos_kf[:, 1], label="KF", color="red", linestyle="--", zorder=3)
        ax_traj.set_title("Trajectory")
        ax_traj.set_xlabel("X Position")
        ax_traj.set_ylabel("Y Position")
        ax_traj.grid(True)
        ax_traj.legend()
        ax_traj.set_xlim(-1, 5)
        ax_traj.set_ylim(-1, 5)

        # Error subplot
        ax_error = fig.add_subplot(1, 2, 2)
        ax_error.plot(time, error_gt, label="GT Error", color='C0')
        ax_error.plot(time, error_kf, label="KF Error", color='red', linestyle="--")
        ax_error.set_title("Error Over Time")
        ax_error.set_xlabel("Time (s)")
        ax_error.set_ylabel("Error (Euclidean Distance)")
        ax_error.grid(True)
        ax_error.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(f"/Users/joefarah/Desktop/Figures/E&C_Final/Task_4/{name}_trajectory_error.png", dpi=300)
        plt.show()


    # Plot comparison of metrics for MPCT and MPCK
    steady_state_error_mpct, settling_time_mpct, overshoot_mpct, undershoot_mpct = zip(*results["MPCT"])
    steady_state_error_mpck, settling_time_mpck, overshoot_mpck, undershoot_mpck = zip(*results["MPCK"])
    test_labels = list(tests.keys())

    # Replace overshoot values of 0 with the corresponding undershoot values
    overshoot_mpct = [-u if o == 0 else o for o, u in zip(overshoot_mpct, undershoot_mpct)]
    overshoot_mpck = [-u if o == 0 else o for o, u in zip(overshoot_mpck, undershoot_mpck)]

    # Bar width for side-by-side bars
    bar_width = 0.35
    x = np.arange(len(test_labels))

    # Plot Steady-State Error
    plt.figure(figsize=(10, 5))
    plt.bar(x - bar_width/2, steady_state_error_mpct, width=bar_width, label='MPCT', color='C0', zorder=3)
    plt.bar(x + bar_width/2, steady_state_error_mpck, width=bar_width, label='MPCK', color='red', zorder=3)
    plt.title("Steady-State Error Comparison")
    plt.xlabel("Test Scenarios")
    plt.ylabel("Steady-State Error")
    plt.xticks(x, test_labels)
    plt.legend()
    plt.grid(axis='y', zorder=-1, alpha=0.5)

    # Add values on top of each bar
    for i, (mpct, mpck) in enumerate(zip(steady_state_error_mpct, steady_state_error_mpck)):
        plt.text(i - bar_width/2, mpct + 0.007, f"{mpct:.2f}", ha='center', color='C0', zorder=4)
        plt.text(i + bar_width/2, mpck + 0.007, f"{mpck:.2f}", ha='center', color='red', zorder=4)

    # Save the figure
    # plt.savefig("/Users/joefarah/Desktop/Figures/E&C_Final/Task_4/mpct_mpck_test_sse.png", dpi=300)
    plt.show()

    # Plot Settling Time
    plt.figure(figsize=(10, 5))
    plt.bar(x - bar_width/2, settling_time_mpct, width=bar_width, label='MPCT', color='C0', zorder=3)
    plt.bar(x + bar_width/2, settling_time_mpck, width=bar_width, label='MPCK', color='red', zorder=3)
    plt.title("Settling Time Comparison")
    plt.xlabel("Test Scenarios")
    plt.ylabel("Settling Time (s)")
    plt.xticks(x, test_labels)
    plt.legend()
    plt.grid(axis='y', zorder=-1, alpha=0.5)

    # Add values on top of each bar
    for i, (mpct, mpck) in enumerate(zip(settling_time_mpct, settling_time_mpck)):
        plt.text(i - bar_width/2, mpct + 0.08, f"{mpct:.2f}", ha='center', color='C0', zorder=4)
        plt.text(i + bar_width/2, mpck + 0.08, f"{mpck:.2f}", ha='center', color='red', zorder=4)

    # Save the figure
    # plt.savefig("/Users/joefarah/Desktop/Figures/E&C_Final/Task_4/mpct_mpck_test_settlingtime.png", dpi=300)
    plt.show()

    # Plot Overshoot
    plt.figure(figsize=(10, 5))
    plt.bar(x - bar_width/2, overshoot_mpct, width=bar_width, label='MPCT', color='C0', zorder=3)
    plt.bar(x + bar_width/2, overshoot_mpck, width=bar_width, label='MPCK', color='red', zorder=3)
    plt.title("Overshoot Comparison")
    plt.xlabel("Test Scenarios")
    plt.ylabel("Overshoot (%)")
    plt.xticks(x, test_labels)
    plt.legend()
    plt.grid(axis='y', zorder=-1, alpha=0.5)
    plt.ylim(-30, 30)

    # Add values on top or bottom of each bar for Overshoot
    for i, (mpct, mpck) in enumerate(zip(overshoot_mpct, overshoot_mpck)):
        # For MPCT bars
        if mpct >= 0:
            plt.text(i - bar_width/2, mpct + 0.6, f"{mpct:.2f}%", ha='center', color='C0', zorder=4)
        else:
            plt.text(i - bar_width/2, mpct - 1.8, f"{mpct:.2f}%", ha='center', color='C0', zorder=4)

        # For MPCK bars
        if mpck >= 0:
            plt.text(i + bar_width/2, mpck + 0.6, f"{mpck:.2f}%", ha='center', color='red', zorder=4)
        else:
            plt.text(i + bar_width/2, mpck - 1.8, f"{mpck:.2f}%", ha='center', color='red', zorder=4)

    # Save the figure
    # plt.savefig("/Users/joefarah/Desktop/Figures/E&C_Final/Task_4/mpct_mpck_test_overshoot.png", dpi=300)
    plt.show()