import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==================================================================
# 1. LOAD DATA (5 COLUMNS)
# ==================================================================
file_path = 'EX3/SLAM.TXT'
try:
    data = np.genfromtxt(file_path, delimiter=',')
    # Robot Path
    raw_rx, raw_ry = data[:, 0], data[:, 1]
    # Wall Map
    raw_wx, raw_wy = data[:, 2], data[:, 3]
    # Markers
    types = data[:, 4]
    
    is_anchor = (types == 1)
    print(f"Loaded {len(data)} points.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# ==================================================================
# 2. IDENTIFY LOOP CLOSURES (Using Anchors)
# ==================================================================
anchors_found_indices = []
loop_closures = []
rounds_completed = 0

# NOTE: Threshold needs to be large enough to catch the drift
threshold = 150.0 
min_index_gap = 100 

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
    print("Forcing manual closure (Start <-> End)...")
    loop_closures.append((anchor_indices[-1], anchor_indices[0]))
    rounds_completed = 1

print(f"Rounds: {rounds_completed} | Closures: {len(loop_closures)}")

# ==================================================================
# 3. GLOBAL OPTIMIZATION (CLOSURE ON MAP)
# ==================================================================
# Pre-calculate the fixed offset vector from Robot to Wall
# We assume this relative vector is constant (Rigid Body)
offset_x = raw_wx - raw_rx
offset_y = raw_wy - raw_ry

def slam_objective(flat_poses, rx, ry, off_x, off_y, closures):
    poses = flat_poses.reshape(-1, 2)
    
    # 1. Odometry Error (Keep the robot path shape stiff)
    diffs = np.diff(poses, axis=0)
    orig_diffs = np.diff(np.column_stack((rx, ry)), axis=0)
    odom_err = np.sum((diffs - orig_diffs)**2)
    
    # 2. Map Closure Error (Force WALLS to overlap)
    # Instead of pulling robot poses together, we pull the calculated wall positions together
    loop_err = 0
    for curr, prev in closures:
        # Calculate where the wall is for the Current optimized pose
        curr_wall_x = poses[curr, 0] + off_x[curr]
        curr_wall_y = poses[curr, 1] + off_y[curr]
        
        # Calculate where the wall is for the Previous optimized pose
        prev_wall_x = poses[prev, 0] + off_x[prev]
        prev_wall_y = poses[prev, 1] + off_y[prev]
        
        # Minimize the distance between the two WALL detections
        dist_sq = (curr_wall_x - prev_wall_x)**2 + (curr_wall_y - prev_wall_y)**2
        loop_err += 50000.0 * dist_sq
        
    return odom_err + loop_err

initial_guess = np.column_stack((raw_rx, raw_ry)).flatten()

# Pass the offsets to the optimizer
res = minimize(slam_objective, initial_guess, 
               args=(raw_rx, raw_ry, offset_x, offset_y, loop_closures), 
               method='L-BFGS-B', options={'maxiter': 2000})

opt_rx = res.x.reshape(-1, 2)[:, 0]
opt_ry = res.x.reshape(-1, 2)[:, 1]

# Recalculate final wall positions based on optimized path
opt_wx = opt_rx + offset_x
opt_wy = opt_ry + offset_y

# ==================================================================
# 4. FILTER & EXPORT
# ==================================================================
start_idx = min([c[1] for c in loop_closures])
end_idx = max([c[0] for c in loop_closures])

# --- SWITCHED TO OPTION A (Save the Corrected Result) ---
final_rx = opt_rx[start_idx:end_idx+1]
final_ry = opt_ry[start_idx:end_idx+1]
final_wx = opt_wx[start_idx:end_idx+1]
final_wy = opt_wy[start_idx:end_idx+1]

output_data = np.column_stack((final_rx, final_ry, final_wx, final_wy))
np.savetxt('EX3/SLAM_CLEANED.TXT', output_data, delimiter=',', fmt='%.2f')

print(f"Exported {len(output_data)} rows to SLAM_CLEANED.TXT")

# ==================================================================
# 5. VISUALIZATION
# ==================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.plot(raw_rx, raw_ry, color='blue', alpha=0.5)
ax1.scatter(raw_wx, raw_wy, color='gray', s=1, alpha=0.3)
ax1.set_title("1. Raw Data")

ax2.plot(raw_rx, raw_ry, color='gray', alpha=0.3)
# Draw lines connecting the WALLS at loop closures to show the new logic
for c in loop_closures:
    ax2.plot([raw_wx[c[0]], raw_wx[c[1]]], [raw_wy[c[0]], raw_wy[c[1]]], 'r--', alpha=0.8, label='Map Connection')
ax2.set_title("2. Closures (On Walls)")

ax3.plot(final_rx, final_ry, color='blue', linewidth=2, label='Path')
ax3.scatter(final_wx, final_wy, color='black', s=5, alpha=0.6, label='Walls')
ax3.set_title("3. Map-Corrected Result")
ax3.legend()

for ax in [ax1, ax2, ax3]:
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()