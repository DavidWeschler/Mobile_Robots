import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==================================================================
# 1. LOAD DATA (5 COLUMNS)
# ==================================================================
file_path = 'EX3/SLAM.TXT'
try:
    # Input Format: RobotX, RobotY, WallX, WallY, Marker
    data = np.genfromtxt(file_path, delimiter=',')
    
    # Extract Robot Trajectory
    raw_rx, raw_ry = data[:, 0], data[:, 1]
    
    # Extract Wall Coordinates
    raw_wx, raw_wy = data[:, 2], data[:, 3]
    
    # Extract Markers
    types = data[:, 4]
    
    is_anchor = (types == 1)
    is_collision = (types == 2)
    print(f"Successfully loaded {len(data)} points with 5 columns.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# ==================================================================
# 2. IDENTIFY LOOP CLOSURES
# ==================================================================
anchors_found_indices = []
loop_closures = []
rounds_completed = 0

threshold = 50.0 
min_index_gap = 100  # Minimum index gap to avoid trivial closures

anchor_indices = np.where(is_anchor)[0]

for i in anchor_indices:
    current_p = np.array([raw_rx[i], raw_ry[i]])
    for prev_idx in anchors_found_indices:
        prev_p = np.array([raw_rx[prev_idx], raw_ry[prev_idx]])
        dist = np.linalg.norm(current_p - prev_p)
        
        if dist < threshold and (i - prev_idx) > min_index_gap:
            loop_closures.append((i, prev_idx))
            if prev_idx == anchor_indices[0]:
                rounds_completed += 1
    
    anchors_found_indices.append(i)

if len(loop_closures) == 0 and len(anchor_indices) >= 2:
    print("No loops found automatically. Forcing first/last closure...")
    loop_closures.append((anchor_indices[-1], anchor_indices[0]))
    rounds_completed = 1

print(f"Detected Rounds: {rounds_completed}")

# ==================================================================
# 3. GLOBAL OPTIMIZATION (Trajectory Only)
# ==================================================================
def slam_objective(flat_poses, rx, ry, closures):
    poses = flat_poses.reshape(-1, 2)
    # Odometry Error
    diffs = np.diff(poses, axis=0)
    orig_diffs = np.diff(np.column_stack((rx, ry)), axis=0)
    odom_err = np.sum((diffs - orig_diffs)**2)
    
    # Loop Closure Error
    loop_err = 0
    for curr, prev in closures:
        loop_err += 50000.0 * np.sum((poses[curr] - poses[prev])**2)
    return odom_err + loop_err

initial_guess = np.column_stack((raw_rx, raw_ry)).flatten()
res = minimize(slam_objective, initial_guess, args=(raw_rx, raw_ry, loop_closures), 
               method='L-BFGS-B', options={'maxiter': 2000})

opt_rx = res.x.reshape(-1, 2)[:, 0]
opt_ry = res.x.reshape(-1, 2)[:, 1]

# ==================================================================
# 4. CORRECT WALL POSITIONS
# ==================================================================
# Calculate the relative offset of the wall from the robot in the RAW data
offset_wx = raw_wx - raw_rx
offset_wy = raw_wy - raw_ry

# Apply that same offset to the NEW optimized robot path
opt_wx = opt_rx + offset_wx
opt_wy = opt_ry + offset_wy

# ==================================================================
# 5. FILTER & EXPORT
# ==================================================================
start_idx = min([c[1] for c in loop_closures])
end_idx = max([c[0] for c in loop_closures])

# --- OPTION A: Save OPTIMIZED (Fixed) Map ---
# final_rx = opt_rx[start_idx:end_idx+1]
# final_ry = opt_ry[start_idx:end_idx+1]
# final_wx = opt_wx[start_idx:end_idx+1]
# final_wy = opt_wy[start_idx:end_idx+1]

# --- OPTION B: Save RAW (Drifted) Map (Truncated) ---
final_rx = raw_rx[start_idx:end_idx+1]
final_ry = raw_ry[start_idx:end_idx+1]
final_wx = raw_wx[start_idx:end_idx+1]
final_wy = raw_wy[start_idx:end_idx+1]

# Stack and Save
output_data = np.column_stack((final_rx, final_ry, final_wx, final_wy))
np.savetxt('EX3/SLAM_CLEANED.TXT', output_data, delimiter=',', fmt='%.2f')

print(f"Exported {len(output_data)} rows. (Type: OPTIMIZED)")

# ==================================================================
# 6. VISUALIZATION
# ==================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Raw Data
ax1.plot(raw_rx, raw_ry, color='blue', alpha=0.5, label='Robot Path')
ax1.scatter(raw_wx, raw_wy, color='gray', s=1, alpha=0.3, label='Walls')
ax1.set_title("1. Raw Data (Drifted)")
ax1.legend()

# Plot 2: Loop Closures
ax2.plot(raw_rx, raw_ry, color='gray', alpha=0.3)
for c in loop_closures:
    ax2.plot([raw_rx[c[0]], raw_rx[c[1]]], [raw_ry[c[0]], raw_ry[c[1]]], 'g--', alpha=0.6)
ax2.set_title(f"2. {len(loop_closures)} Loop Closures")

# Plot 3: Final Result (Path + Walls)
ax3.plot(final_rx, final_ry, color='blue', linewidth=2, label='Corrected Path')
ax3.scatter(final_wx, final_wy, color='black', s=5, alpha=0.6, label='Corrected Walls')
ax3.set_title("3. Final SLAM Map")
ax3.legend()

for ax in [ax1, ax2, ax3]:
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()