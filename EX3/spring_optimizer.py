import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Load your data (x, y, is_anchor)
data = np.genfromtxt('SLAM_DATA.TXT', delimiter=',')
raw_x, raw_y = data[:, 0], data[:, 1]
is_anchor = data[:, 2] == 1

# 2. Identify Loop Closures
# We find indices where the robot saw an anchor it had seen before
anchors_found = []
loop_closures = [] # Stores (current_index, first_seen_index)

threshold = 20.0 # 20cm radius to consider it the "same" anchor
for i in range(len(data)):
    if is_anchor[i]:
        current_p = np.array([raw_x[i], raw_y[i]])
        match_found = False
        for prev_idx in anchors_found:
            prev_p = np.array([raw_x[prev_idx], raw_y[prev_idx]])
            if np.linalg.norm(current_p - prev_p) < threshold:
                loop_closures.append((i, prev_idx))
                match_found = True
                break
        if not match_found:
            anchors_found.append(i)

# 3. The "Spring" Optimization (Objective Function)
def objective_function(flat_poses, raw_x, raw_y, loop_closures):
    poses = flat_poses.reshape(-1, 2)
    error = 0
    
    # Odometry Springs: Keep points near their original relative distance
    # Stiffness (Weight) for odometry is 1.0
    for i in range(1, len(poses)):
        actual_dist = poses[i] - poses[i-1]
        original_dist = np.array([raw_x[i]-raw_x[i-1], raw_y[i]-raw_y[i-1]])
        error += np.sum((actual_dist - original_dist)**2)
        
    # Loop Closure Springs: Pull matching anchors together
    # Stiffness is much higher (10.0) because we know these are the same point
    for curr_idx, prev_idx in loop_closures:
        dist = poses[curr_idx] - poses[prev_idx]
        error += 10.0 * np.sum(dist**2)
        
    return error

# 4. Run Optimization
initial_guess = np.column_stack((raw_x, raw_y)).flatten()
res = minimize(objective_function, initial_guess, args=(raw_x, raw_y, loop_closures))
optimized_poses = res.x.reshape(-1, 2)

# 5. Visual Comparison
plt.figure(figsize=(12, 6))
plt.plot(raw_x, raw_y, 'r--', label='Raw Odometry (Drifted)', alpha=0.5)
plt.plot(optimized_poses[:,0], optimized_poses[:,1], 'b-', label='Optimized (Loop Closure)')
plt.scatter(optimized_poses[is_anchor, 0], optimized_poses[is_anchor, 1], color='black', label='Anchors')
plt.legend()
plt.title("Spring-Based Loop Closure (Pose Graph Optimization)")
plt.show()