import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import minimize

# ==================================================================
# 1. CONFIGURATION
# ==================================================================
# FILE_PATH = r'C:\Users\isrgd\robots\Mobile_Robots\EX3\TRACK_LOG_de_best.TXT'
# FILE_PATH = r'C:\Users\isrgd\robots\Mobile_Robots\EX3\TRACK_LOG.TXT'
# FILE_PATH = r'C:\Users\isrgd\robots\Mobile_Robots\EX3\TRACK_LOG_11_M.TXT'
# FILE_PATH = r'C:\Users\isrgd\robots\Mobile_Robots\EX3\TRACK_LOG_18_M.TXT'

# Scaling factors (Matches your NXC code)
SCALE_XY = 10000.0
SCALE_THETA = 100000.0

# Sensor Geometry (90 degrees to the left)
SENSOR_ANGLE_OFFSET = math.pi / 2.0

# Optimization Settings
SPRING_STIFFNESS = 5000000.0  # How hard we pull the loop closed

# RANSAC Settings
RANSAC_THRESHOLD = 10.0     # Max distance (cm) for a point to fit a line
RANSAC_MIN_POINTS = 10      # Minimum points to call it a "wall"

# Fix random seed for consistent results
random.seed(42)
np.random.seed(42)

# ==================================================================
# 2. HELPER FUNCTIONS
# ==================================================================
def calculate_wall_coords(rx, ry, th, dist):
    """
    Calculates absolute wall coordinates based on robot pose and sensor reading.
    Wall = Robot + (Dist * angle_vector)
    """
    # Angle of the sensor in the global frame
    sensor_global_angle = th + SENSOR_ANGLE_OFFSET
    
    wx = rx + (dist * np.cos(sensor_global_angle))
    wy = ry + (dist * np.sin(sensor_global_angle))
    return wx, wy

def fit_line(x, y):
    """Fits a line to points using Linear Least Squares."""
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# ==================================================================
# 3. SPRING OPTIMIZATION (SLAM)
# ==================================================================
def slam_objective_function(flat_poses, raw_rx, raw_ry, closures):
    """
    The 'Spring' Physics Engine.
    1. Odometry Springs: Keep the path shape rigid (minimize change from raw).
    2. Closure Springs: Pull the start and end points together.
    """
    # Reshape the flat array back into (N, 2)
    poses = flat_poses.reshape(-1, 2)
    
    # --- Constraint A: Odometry (Maintain Shape) ---
    # Calculate the vector differences between consecutive points (the "steps")
    current_steps = np.diff(poses, axis=0)
    original_steps = np.diff(np.column_stack((raw_rx, raw_ry)), axis=0)
    
    # We want the new steps to be as close as possible to the original steps
    odometry_error = np.sum((current_steps - original_steps)**2)
    
    # --- Constraint B: Loop Closure (Close the Gap) ---
    closure_error = 0
    for idx_a, idx_b in closures:
        # Distance between the two anchor points (should be 0)
        dist_sq = np.sum((poses[idx_a] - poses[idx_b])**2)
        closure_error += SPRING_STIFFNESS * dist_sq
        
    return odometry_error + closure_error

# ==================================================================
# 4. MAIN PROCESSING
# ==================================================================
def main():
    print("--- SLAM PROCESS START ---")

    # --- STEP 1: LOAD & SCALE DATA ---
    try:
        data = np.genfromtxt(FILE_PATH, delimiter=',')
        # Col 0: X, Col 1: Y, Col 2: Theta, Col 3: Dist, Col 4: Marker
        
        # Apply Scaling immediately
        raw_rx = data[:, 0] / SCALE_XY
        raw_ry = data[:, 1] / SCALE_XY
        raw_th = data[:, 2] / SCALE_THETA
        raw_dist = data[:, 3]
        markers = data[:, 4]
        
        print(f"Loaded {len(data)} points.")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- STEP 2: TRIM TO LOOP (Start Anchor -> End Anchor) ---
    anchors = np.where(markers == 1)[0]
    
    if len(anchors) < 2:
        print("Warning: Less than 2 anchors found. Using full path.")
        start_idx, end_idx = 0, len(data) - 1
    else:
        start_idx, end_idx = anchors[0], anchors[-1]
        print(f"Trimming data: Rows {start_idx} to {end_idx}")

    # Slice the arrays
    rx = raw_rx[start_idx:end_idx+1]
    ry = raw_ry[start_idx:end_idx+1]
    th = raw_th[start_idx:end_idx+1]
    dist = raw_dist[start_idx:end_idx+1]

    # --- STEP 3: RUN OPTIMIZATION ---
    print("Optimizing Path (Spring Physics)...")
    
    # Define constraints: Connect Last Point -> First Point
    closures = [(len(rx)-1, 0)] 
    
    # Initial guess is just the raw path
    initial_guess = np.column_stack((rx, ry)).flatten()
    
    result = minimize(slam_objective_function, initial_guess, 
                      args=(rx, ry, closures),
                      method='L-BFGS-B', 
                      options={'maxiter': 1000, 'disp': True})
    
    # Extract corrected path
    opt_rx = result.x.reshape(-1, 2)[:, 0]
    opt_ry = result.x.reshape(-1, 2)[:, 1]

    # --- STEP 4: CALCULATE WALLS (Based on Corrected Path) ---
    print("Calculating Wall Positions...")
    # Note: We use the ORIGINAL Theta and Distance, but apply them to the NEW X,Y
    # This assumes the robot's heading error was small compared to position error.
    opt_wx, opt_wy = calculate_wall_coords(opt_rx, opt_ry, th, dist)

    # --- STEP 5: RANSAC (Find Lines) ---
    print("Running RANSAC...")
    
    wall_points_x = list(opt_wx)
    wall_points_y = list(opt_wy)
    detected_lines = []
    
    # Keep finding lines until we run out of points
    while len(wall_points_x) > RANSAC_MIN_POINTS:
        best_inliers = []
        best_model = None
        
        # Iterations for current line
        for _ in range(200):
            # Pick 2 random points
            idx = random.sample(range(len(wall_points_x)), 2)
            p1 = (wall_points_x[idx[0]], wall_points_y[idx[0]])
            p2 = (wall_points_x[idx[1]], wall_points_y[idx[1]])
            
            # Form Line Equation: Ax + By + C = 0
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = -A*p1[0] - B*p1[1]
            denom = np.sqrt(A*A + B*B)
            if denom == 0: continue

            # Count inliers
            current_inliers = []
            for i in range(len(wall_points_x)):
                d = abs(A*wall_points_x[i] + B*wall_points_y[i] + C) / denom
                if d < RANSAC_THRESHOLD:
                    current_inliers.append(i)
            
            if len(current_inliers) > len(best_inliers):
                best_inliers = current_inliers
                best_model = (A, B, C)
        
        # If we found a good line
        if len(best_inliers) > RANSAC_MIN_POINTS:
            # Extract inlier points
            lx = [wall_points_x[i] for i in best_inliers]
            ly = [wall_points_y[i] for i in best_inliers]
            
            detected_lines.append((lx, ly))
            
            # Remove these points from the dataset
            wall_points_x = [wall_points_x[i] for i in range(len(wall_points_x)) if i not in best_inliers]
            wall_points_y = [wall_points_y[i] for i in range(len(wall_points_y)) if i not in best_inliers]
        else:
            break # No more valid lines found

    # ==================================================================
    # 5. PLOTTING
    # ==================================================================
    print(f"Plotting results... Found {len(detected_lines)} walls.")
    
    plt.figure(figsize=(12, 6))
    
    # --- SUBPLOT 1: Raw Data (Drifted) ---
    plt.subplot(1, 2, 1)
    # Calculate raw walls for visualization
    raw_wx, raw_wy = calculate_wall_coords(rx, ry, th, dist)
    
    plt.plot(rx, ry, 'b-', alpha=0.5, label='Robot Path')
    plt.scatter(raw_wx, raw_wy, c='gray', s=2, alpha=0.3, label='Sensor Hits')
    
    # Mark Start/End
    plt.scatter(rx[0], ry[0], c='g', marker='o', s=50, label='Start')
    plt.scatter(rx[-1], ry[-1], c='r', marker='x', s=50, label='End')
    
    plt.title("1. Raw Data (Drifted)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    # --- SUBPLOT 2: Optimized & RANSAC ---
    plt.subplot(1, 2, 2)
    
    # 1. Plot Corrected Robot Path
    plt.plot(opt_rx, opt_ry, 'b-', linewidth=2, label='Optimized Path')
    
    # 2. Plot Raw Wall Points (Corrected Position)
    plt.scatter(opt_wx, opt_wy, c='gray', s=1, alpha=0.2)
    
    # 3. Plot RANSAC Lines
    colors = ['r', 'g', 'm', 'c', 'y', 'k']
    for i, (lx, ly) in enumerate(detected_lines):
        c = colors[i % len(colors)]
        # Draw the line segment from min to max
        # Simple fit for drawing
        m, c_val = fit_line(lx, ly)
        x_min, x_max = min(lx), max(lx)
        y_min = m*x_min + c_val
        y_max = m*x_max + c_val
        
        plt.plot([x_min, x_max], [y_min, y_max], color=c, linewidth=3, label=f'Wall {i+1}')
        
    plt.title(f"2. Final Map ({len(detected_lines)} Walls)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()