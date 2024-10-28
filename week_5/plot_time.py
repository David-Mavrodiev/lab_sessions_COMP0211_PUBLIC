import matplotlib.pyplot as plt
import numpy as np

# Data: time taken for different values of amplitude (P_w) and frequency (V_w)
time = {(1000, 1): 15.97, (10000, 10): 18.62, (100000, 100): 25.12}

# Create a figure for plotting
plt.figure()

# Define a colormap for the scatter points
colors = plt.cm.viridis(np.linspace(0, 1, len(time)))

# Prepare lists to store labels and times
p_w_values = []
times = []

# Iterate over the time dictionary and plot each point
for i, ((p_w, v_w), t) in enumerate(time.items()):
    # Combine p_w and v_w into a label with LaTeX formatting
    combined_label = rf'${p_w}, {v_w}$'
    
    # Scatter the points
    plt.scatter(combined_label, t, color=colors[i], label=rf'$P_w={p_w}, V_w={v_w}, \mathrm{{time}}={t:.2f}\,s$')
    
    # Append to lists for plotting
    p_w_values.append(combined_label)
    times.append(t)

# Plot the points and connect them with a dashed line
plt.plot(p_w_values, times, color='gray', linestyle='--', linewidth=1)

# Add labels and title with LaTeX formatting
plt.xlabel(r'$P_w, V_w$')
plt.ylabel(r'Time (s)')
plt.title(r'Time taken for different values of $P_w$ and $V_w$')

# Show the legend
plt.legend()

# Display the grid and plot
plt.grid(True)
# plt.savefig("/Users/joefarah/Desktop/Figures/E&C_Lab3/time_plot.png", dpi=300)
plt.show()