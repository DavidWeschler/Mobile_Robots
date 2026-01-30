import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==================================================================
# 1. LOAD DATA
# ==================================================================
file_path = 'EX3/SLAM.TXT'
try:
    # Format: x, y, marker (0=path, 1=anchor, 2=collision)
    data = np.genfromtxt(file_path, delimiter=',')
    raw_x, raw_y, types = data[:, 0], data[:, 1], data[:, 2]
    is_anchor = (types == 1)
    is_collision = (types == 2)
    print(f"Successfully loaded {len(data)} points from {file_path}")
except Exception as e:
    print(f"Error loading file: {e}. Check if the file exists in the directory.")
    exit()

# ==================================================================
# 2. IDENTIFY LOOP CLOSURES & ROUNDS
# ==================================================================
anchors_found_indices = []
loop_closures = []
rounds_completed = 0

# Adjusted threshold to handle large drifts (~150cm)
threshold = 150.0 
min_index_gap = 50 # Prevents matching the same dot immediately

anchor_indices = np.where(is_anchor)[0]

for i in anchor_indices:
    current_p = np.array([raw_x[i], raw_y[i]])
    for prev_idx in anchors_found_indices:
        prev_p = np.array([raw_x[prev_idx], raw_y[prev_idx]])
        dist = np.linalg.norm(current_p - prev_p)
        
        if dist < threshold and (i - prev_idx) > min_index_gap:
            loop_closures.append((i, prev_idx))
            # If we returned to the very first anchor seen, count a round
            if prev_idx == anchor_indices[0]:
                rounds_completed += 1
    
    anchors_found_indices.append(i)

# FORCE CLOSURE FALLBACK: If no loops found, match the last anchor to the first
if len(loop_closures) == 0 and len(anchor_indices) >= 2:
    print("No loops found automatically. Forcing first/last closure...")
    loop_closures.append((anchor_indices[-1], anchor_indices[0]))
    rounds_completed = 1

print(f"Detected Rounds: {rounds_completed}")
print(f"Total Loop Closures (Springs): {len(loop_closures)}")

# ==================================================================
# 3. GLOBAL OPTIMIZATION (THE "FIX")
# ==================================================================
def slam_objective(flat_poses, rx, ry, closures):
    poses = flat_poses.reshape(-1, 2)
    # Odometry Error (Stiffness 1.0)
    diffs = np.diff(poses, axis=0)
    orig_diffs = np.diff(np.column_stack((rx, ry)), axis=0)
    odom_err = np.sum((diffs - orig_diffs)**2)
    
    # Loop Closure Error (Stiffness 50,000)
    loop_err = 0
    for curr, prev in closures:
        loop_err += 50000.0 * np.sum((poses[curr] - poses[prev])**2)
    return odom_err + loop_err

initial_guess = np.column_stack((raw_x, raw_y)).flatten()
res = minimize(slam_objective, initial_guess, args=(raw_x, raw_y, loop_closures), 
               method='L-BFGS-B', options={'maxiter': 2000})

optimized_poses = res.x.reshape(-1, 2)

# ==================================================================
# 4. FILTER BOUNDS & EXPORT FILES
# ==================================================================
# Identify the start of the first closure and end of the last closure
start_idx = min([c[1] for c in loop_closures])
end_idx = max([c[0] for c in loop_closures])

# --- FILE A: SLAM_CLEANED.TXT (The Fixed Map) ---
final_x = optimized_poses[start_idx:end_idx+1, 0]
final_y = optimized_poses[start_idx:end_idx+1, 1]
final_types = types[start_idx:end_idx+1]

output_cleaned = np.column_stack((final_x, final_y, final_types))
np.savetxt('EX3/SLAM_CLEANED.TXT', output_cleaned, delimiter=',', fmt='%.2f,%.2f,%d')

# --- FILE B: RAW_LAP_ONLY.TXT (Plot 2 Style - No points outside anchors) ---
raw_lap_x = raw_x[start_idx:end_idx+1]
raw_lap_y = raw_y[start_idx:end_idx+1]
raw_lap_types = types[start_idx:end_idx+1]

output_raw_lap = np.column_stack((raw_lap_x, raw_lap_y, raw_lap_types))
np.savetxt('EX3/RAW_LAP_ONLY.TXT', output_raw_lap, delimiter=',', fmt='%.2f,%.2f,%d')

print(f"Exported SLAM_CLEANED.TXT and RAW_LAP_ONLY.TXT ({len(output_cleaned)} points each).")

# ==================================================================
# 5. VISUALIZATION
# ==================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# PLOT 1: FULL RAW DATA
ax1.plot(raw_x, raw_y, color='gray', alpha=0.5)
ax1.scatter(raw_x[0], raw_y[0], color='green', marker='P', s=100, label='Start')
ax1.set_title("1. Full Raw Recording")

# PLOT 2: TRUNCATED RAW WITH CONNECTIONS
ax2.plot(raw_lap_x, raw_lap_y, color='gray', alpha=0.2)
ax2.scatter(raw_lap_x[raw_lap_types==1], raw_lap_y[raw_lap_types==1], c='blue', s=15)
for c in loop_closures:
    # Adjust closure indices for the truncated plot
    ax2.plot([raw_x[c[0]], raw_x[c[1]]], [raw_y[c[0]], raw_y[c[1]]], 'g--', alpha=0.6)
ax2.set_title(f"2. Truncated Raw & {rounds_completed} Rounds")

# PLOT 3: FINAL SLAM RESULT
ax3.plot(final_x, final_y, color='blue', linewidth=2, label='Optimized Loop')
ax3.scatter(final_x[final_types==1], final_y[final_types==1], c='black', marker='*', s=50, label='Anchors')
if any(final_types == 2):
    ax3.scatter(final_x[final_types==2], final_y[final_types==2], c='red', marker='x', label='Collisions')
ax3.set_title("3. Final SLAM Result")
ax3.legend()

for ax in [ax1, ax2, ax3]:
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()