import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==================================================================
# 1. LOAD DATA
# ==================================================================
try:
    data = np.genfromtxt('EX3/SLAM.TXT', delimiter=',')
    raw_x, raw_y, types = data[:, 0], data[:, 1], data[:, 2]
    is_anchor = (types == 1)
    is_collision = (types == 2)
    print(f"Loaded {len(data)} points.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# ==================================================================
# 2. IDENTIFY LOOP CLOSURES
# ==================================================================
anchors_found_indices = []
loop_closures = []
threshold = 150.0 
min_index_gap = 200 

anchor_indices = np.where(is_anchor)[0]

for i in anchor_indices:
    current_p = np.array([raw_x[i], raw_y[i]])
    for prev_idx in anchors_found_indices:
        prev_p = np.array([raw_x[prev_idx], raw_y[prev_idx]])
        dist = np.linalg.norm(current_p - prev_p)
        if dist < threshold and (i - prev_idx) > min_index_gap:
            loop_closures.append((i, prev_idx))
    anchors_found_indices.append(i)

# FORCE CLOSURE if none found
if len(loop_closures) == 0 and len(anchor_indices) >= 2:
    loop_closures.append((anchor_indices[-1], anchor_indices[0]))

print(f"Total Loop Closures: {len(loop_closures)}")

# ==================================================================
# 3. OPTIMIZATION
# ==================================================================
def slam_objective(flat_poses, rx, ry, closures):
    poses = flat_poses.reshape(-1, 2)
    diffs = np.diff(poses, axis=0)
    orig_diffs = np.diff(np.column_stack((rx, ry)), axis=0)
    odom_err = np.sum((diffs - orig_diffs)**2)
    loop_err = sum(50000.0 * np.sum((poses[c[0]] - poses[c[1]])**2) for c in closures)
    return odom_err + loop_err

initial_guess = np.column_stack((raw_x, raw_y)).flatten()
res = minimize(slam_objective, initial_guess, args=(raw_x, raw_y, loop_closures), 
               method='L-BFGS-B', options={'maxiter': 2000})
optimized_poses = res.x.reshape(-1, 2)

# ==================================================================
# 4. FILTER AND SAVE (points inside the loop only)
# ==================================================================
# We find the start of the first closure and end of the last closure
start_idx = min([c[1] for c in loop_closures])
end_idx = max([c[0] for c in loop_closures])

# Slice the optimized data
final_x = optimized_poses[start_idx:end_idx+1, 0]
final_y = optimized_poses[start_idx:end_idx+1, 1]
final_types = types[start_idx:end_idx+1]

# Combine into original format
output_data = np.column_stack((final_x, final_y, final_types))

# Save to file
np.savetxt('EX3/SLAM_CLEANED.TXT', output_data, delimiter=',', fmt='%.2f,%.2f,%d')
print(f"Saved {len(output_data)} points to SLAM_CLEANED.TXT (Points outside loop removed)")

# ==================================================================
# 5. VISUALIZATION
# ==================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.plot(raw_x, raw_y, color='gray', alpha=0.5)
ax1.set_title("1. Raw Data")

ax2.plot(raw_x, raw_y, color='gray', alpha=0.2)
for c in loop_closures:
    ax2.plot([raw_x[c[0]], raw_x[c[1]]], [raw_y[c[0]], raw_y[c[1]]], 'g--')
ax2.set_title("2. Closures Found")

ax3.plot(final_x, final_y, color='blue', label='Cleaned Loop')
ax3.scatter(final_x[final_types==1], final_y[final_types==1], c='black', marker='*')
ax3.set_title("3. Final Cleaned Result")

for ax in [ax1, ax2, ax3]:
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()