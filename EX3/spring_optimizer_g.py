import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import minimize

# ==================================================================
# CONFIGURATION
# ==================================================================
FILE_PATH = r'C:\Users\isrgd\robots\Mobile_Robots\EX3\TRACK_LOG_de_best.TXT'

# --- SCALES (MATCHING YOUR NXC CODE) ---
# NXC: long xo = logX[i] * 10000;
X_Y_SCALE = 10000.0       

# NXC: long to = logTheta[i] * 100000;
THETA_SCALE = 100000.0     

# NXC: Wall following on LEFT side -> Sensor is at +90 degrees (+pi/2)
SENSOR_OFFSET_ANGLE = math.pi / 2.0  

# Optimization Settings
SPRING_STIFFNESS = 50000.0
# RANSAC Settings
RANSAC_DIST_THRESH = 10.0  
RANSAC_MIN_POINTS = 5

random.seed(42)
np.random.seed(42)

# ==================================================================
# 1. HELPER FUNCTIONS
# ==================================================================
def fit_line_pseudo_inverse(x_points, y_points):
    x = np.array(x_points).reshape(-1, 1)
    y = np.array(y_points).reshape(-1, 1)
    if len(x) < 2: return 0, 0
    if np.std(x) < 0.01 * np.std(y): return float('inf'), np.mean(x)
    
    A = np.hstack([x, np.ones_like(x)])
    try:
        p = np.linalg.pinv(A) @ y
        return p[0][0], p[1][0]
    except:
        return 0, 0

def extract_walls_ransac(x_all, y_all):
    remaining_indices = list(range(len(x_all)))
    found_walls = []
    
    while len(remaining_indices) > RANSAC_MIN_POINTS:
        best_inliers = []
        # Try 500 random lines
        for _ in range(500):
            if len(remaining_indices) < 2: break
            idx = random.sample(remaining_indices, 2)
            p1 = (x_all[idx[0]], y_all[idx[0]])
            p2 = (x_all[idx[1]], y_all[idx[1]])
            
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = -A * p1[0] - B * p1[1]
            denom = math.sqrt(A*A + B*B)
            if denom == 0: continue
            
            current_inliers = []
            for i in remaining_indices:
                dist = abs(A*x_all[i] + B*y_all[i] + C) / denom
                if dist < RANSAC_DIST_THRESH:
                    current_inliers.append(i)
            
            if len(current_inliers) > len(best_inliers):
                best_inliers = current_inliers

        if len(best_inliers) < RANSAC_MIN_POINTS: break
        
        wx = [x_all[i] for i in best_inliers]
        wy = [y_all[i] for i in best_inliers]
        m, c = fit_line_pseudo_inverse(wx, wy)
        found_walls.append({'m': m, 'c': c, 'x': wx, 'y': wy})
        remaining_indices = [i for i in remaining_indices if i not in best_inliers]
        
    return found_walls

# ==================================================================
# 2. OPTIMIZATION
# ==================================================================
def slam_objective(flat_poses, rx, ry, off_x, off_y, closures):
    poses = flat_poses.reshape(-1, 2)
    # 1. Odometry Error
    diffs = np.diff(poses, axis=0)
    orig_diffs = np.diff(np.column_stack((rx, ry)), axis=0)
    odom_err = np.sum((diffs - orig_diffs)**2)
    
    # 2. Map Closure Error
    loop_err = 0
    for curr, prev in closures:
        cx = poses[curr, 0] + off_x[curr]
        cy = poses[curr, 1] + off_y[curr]
        px = poses[prev, 0] + off_x[prev]
        py = poses[prev, 1] + off_y[prev]
        loop_err += SPRING_STIFFNESS * ((cx-px)**2 + (cy-py)**2)
        
    return odom_err + loop_err

# ==================================================================
# 3. MAIN PIPELINE
# ==================================================================
def main():
    try:
        # Load Data: RobotX, RobotY, Theta, Distance, Marker
        data = np.genfromtxt(FILE_PATH, delimiter=',')
        
        # APPLY NXC SCALING HERE
        full_rx = data[:, 0] / X_Y_SCALE
        full_ry = data[:, 1] / X_Y_SCALE
        full_th = data[:, 2] / THETA_SCALE
        full_dst = data[:, 3]
        full_types = data[:, 4]
        
        print(f"Loaded {len(data)} rows.")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- TRIM DATA ---
    anchors = np.where(full_types == 1)[0]
    if len(anchors) >= 2:
        start, end = anchors[0], anchors[-1]
        print(f"Trimming to rows {start}-{end}")
    else:
        start, end = 0, len(data)-1
        print("Warning: Using full data.")

    rx = full_rx[start:end+1]
    ry = full_ry[start:end+1]
    th = full_th[start:end+1]
    dst = full_dst[start:end+1]

    # --- CALCULATE WALLS ---
    # Wall = Robot + Dist * Angle_Vector
    raw_wx = rx + (dst * np.cos(th + SENSOR_OFFSET_ANGLE))
    raw_wy = ry + (dst * np.sin(th + SENSOR_OFFSET_ANGLE))
    
    # Calculate rigid offsets (Wall relative to Robot)
    offset_x = raw_wx - rx
    offset_y = raw_wy - ry

    # --- OPTIMIZATION ---
    closures = []
    if len(anchors) >= 2:
        closures.append((len(rx)-1, 0)) # Force Close Loop
    
    print(f"Optimizing with {len(closures)} closures...")
    
    initial_guess = np.column_stack((rx, ry)).flatten()
    res = minimize(slam_objective, initial_guess, 
                   args=(rx, ry, offset_x, offset_y, closures), 
                   method='L-BFGS-B', options={'maxiter': 2000})

    opt_rx = res.x.reshape(-1, 2)[:, 0]
    opt_ry = res.x.reshape(-1, 2)[:, 1]
    
    # Recalculate Optimized Wall Positions
    opt_wx = opt_rx + offset_x
    opt_wy = opt_ry + offset_y
    
    # Save Cleaned Data
    out = np.column_stack((opt_rx, opt_ry, opt_wx, opt_wy))
    np.savetxt('SLAM_CLEANED.TXT', out, delimiter=',', fmt='%.2f')
    print("Saved SLAM_CLEANED.TXT")

    # --- RANSAC & PLOT ---
    print("Extracting Walls using RANSAC...")
    walls = extract_walls_ransac(opt_wx, opt_wy)
    print(f"Found {len(walls)} walls.")

    if not walls:
        print("No walls found! Check RANSAC threshold.")
        return

    # Sort and Plot
    all_x = [p for w in walls for p in w['x']]
    all_y = [p for w in walls for p in w['y']]
    cx, cy = np.mean(all_x), np.mean(all_y)
    for w in walls:
        w['angle'] = math.atan2(np.mean(w['y']) - cy, np.mean(w['x']) - cx)
    sorted_walls = sorted(walls, key=lambda w: w['angle'])

    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_walls)))
    
    for i, w in enumerate(sorted_walls):
        plt.plot(w['x'], w['y'], '.', color=colors[i], markersize=3)
        if w['m'] != float('inf'):
            x_range = np.linspace(min(w['x']), max(w['x']), 10)
            plt.plot(x_range, w['m']*x_range + w['c'], '-', color=colors[i], linewidth=2)
        else:
            plt.vlines(w['c'], min(w['y']), max(w['y']), colors[i], linewidth=2)

    plt.title(f"Final Corrected Map ({len(walls)} Walls)")
    plt.axis('equal')
    plt.grid(True, alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()