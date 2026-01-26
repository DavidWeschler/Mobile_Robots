import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Load and Downsample
data = np.genfromtxt('EX3/SLAM.TXT', delimiter=',')
# Only take every 5th point to speed up optimization, but KEEP all anchors
is_anchor = data[:, 2] == 1
indices_to_keep = np.where(is_anchor | (np.arange(len(data)) % 1 == 0))[0]

reduced_data = data[indices_to_keep]
raw_x = reduced_data[:, 0]
raw_y = reduced_data[:, 1]
anchors_mask = reduced_data[:, 2] == 1

# 2. Identify Loop Closures (Radius Search)
anchors_found = []
loop_closures = []
threshold = 10.0 # 25cm radius

for i in range(len(reduced_data)):
    if anchors_mask[i]:
        current_p = np.array([raw_x[i], raw_y[i]])
        match_found = False
        for prev_idx in anchors_found:
            prev_p = np.array([raw_x[prev_idx], raw_y[prev_idx]])
            if np.linalg.norm(current_p - prev_p) < threshold:
                if abs(i - prev_idx) > 50: # Avoid matching the same dot twice in a row
                    loop_closures.append((i, prev_idx))
                    match_found = True
                    break
        if not match_found:
            anchors_found.append(i)

# 3. Fast Optimization Function
def fast_objective(flat_poses, raw_x, raw_y, loop_closures):
    poses = flat_poses.reshape(-1, 2)
    
    # Vectorized Odometry Error (Much faster than loops)
    diffs = np.diff(poses, axis=0)
    orig_diffs = np.diff(np.column_stack((raw_x, raw_y)), axis=0)
    odom_error = np.sum((diffs - orig_diffs)**2)
    
    # Loop Closure Error
    loop_error = 0
    for curr, prev in loop_closures:
        loop_error += 50.0 * np.sum((poses[curr] - poses[prev])**2)
        
    return odom_error + loop_error

# 4. Minimize using 'L-BFGS-B' (Faster for large datasets)
initial_guess = np.column_stack((raw_x, raw_y)).flatten()
res = minimize(fast_objective, initial_guess, 
               args=(raw_x, raw_y, loop_closures), 
               method='L-BFGS-B', 
               options={'maxiter': 1000})

optimized_poses = res.x.reshape(-1, 2)

# 5. Result
plt.figure(figsize=(10, 6))
plt.plot(raw_x, raw_y, 'r--', label='Drifted Path', alpha=0.3)
plt.plot(optimized_poses[:,0], optimized_poses[:,1], 'b-', label='SLAM Corrected')
plt.scatter(optimized_poses[anchors_mask, 0], optimized_poses[anchors_mask, 1], c='black', label='Anchors')
plt.legend()
plt.axis('equal')
plt.show()