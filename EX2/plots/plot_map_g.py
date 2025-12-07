import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = "EX2/plots/MAP.TXT"
X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) 

# Fix randomness to ensure consistent results on every run
random.seed(42)
np.random.seed(42)

# --- 1. Drift Correction ---

def correct_odometry_drift(x_robot, y_robot):
    """
    Corrects odometry drift by forcing the last point to match the first point.
    The error is distributed linearly across all points.
    """
    drift_x = x_robot[-1] - x_robot[0]
    drift_y = y_robot[-1] - y_robot[0]
    num_points = len(x_robot)
    
    x_corrected = np.copy(x_robot)
    y_corrected = np.copy(y_robot)
    
    for i in range(num_points):
        # Calculate correction factor (0.0 at start, 1.0 at end)
        factor = i / (num_points - 1)
        
        # Apply correction (subtract the accumulated drift)
        x_corrected[i] -= drift_x * factor
        y_corrected[i] -= drift_y * factor
        
    return x_corrected, y_corrected

# --- 2. Math Helpers ---

def get_line_params_from_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -A * x1 - B * y1
    return A, B, C

def get_point_line_distances(x_vals, y_vals, A, B, C):
    x = np.array(x_vals)
    y = np.array(y_vals)
    numerator = np.abs(A * x + B * y + C)
    denominator = np.sqrt(A**2 + B**2)
    if denominator == 0: return np.full_like(x, np.inf)
    return numerator / denominator

def fit_line_pseudo_inverse(x_points, y_points):
    x = np.array(x_points).reshape(-1, 1)
    y = np.array(y_points).reshape(-1, 1)
    if np.std(x) < 0.01 * np.std(y): return float('inf'), np.mean(x)
    A = np.hstack([x, np.ones_like(x)])
    A_pinv = np.linalg.pinv(A)
    p = A_pinv @ y
    return p[0][0], p[1][0]

def find_intersection(m1, c1, m2, c2):
    # Handle parallel or vertical lines
    if m1 == float('inf') and m2 == float('inf'): return None
    if m1 == float('inf'): return c1, m2 * c1 + c2
    if m2 == float('inf'): return c2, m1 * c2 + c1
    if abs(m1 - m2) < 1e-5: return None # Parallel
    
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

def extract_walls_ransac(x_all, y_all, max_distance=4.0, min_points=10, iterations=500):
    remaining_indices = list(range(len(x_all)))
    found_walls = []
    
    while len(remaining_indices) > min_points:
        best_inliers_indices = []
        # RANSAC attempts to find a single line
        for _ in range(iterations):
            if len(remaining_indices) < 2: break
            sample_idx = random.sample(remaining_indices, 2)
            p1 = (x_all[sample_idx[0]], y_all[sample_idx[0]])
            p2 = (x_all[sample_idx[1]], y_all[sample_idx[1]])
            
            A, B, C = get_line_params_from_2_points(p1, p2)
            if A == 0 and B == 0: continue
            
            curr_x = [x_all[i] for i in remaining_indices]
            curr_y = [y_all[i] for i in remaining_indices]
            dists = get_point_line_distances(curr_x, curr_y, A, B, C)
            
            # Count inliers (points close to the line)
            current_inliers = []
            for i, d in enumerate(dists):
                if d < max_distance:
                    current_inliers.append(remaining_indices[i])
            
            if len(current_inliers) > len(best_inliers_indices):
                best_inliers_indices = current_inliers
        
        #if len(best_inliers_indices) < min_points: break
        
        # Save the found wall
        wall_x = [x_all[i] for i in best_inliers_indices]
        wall_y = [y_all[i] for i in best_inliers_indices]
        m, c = fit_line_pseudo_inverse(wall_x, wall_y)
        
        found_walls.append({
            'm': m, 'c': c,
            'x_points': wall_x, 'y_points': wall_y
        })
        
        # Remove inliers from the pool
        remaining_indices = [idx for idx in remaining_indices if idx not in best_inliers_indices]
        
    return found_walls

# --- 3. Main Logic (with Geometric Sorting) ---

def process_map_data(x_robot, y_robot, theta_robot, distance_measured):
    
    # 1. Loop Closure Correction
    print("Correcting Drift...")
    x_robot_corr, y_robot_corr = correct_odometry_drift(x_robot, y_robot)
    
    # 2. Recalculate Wall Points
    x_wall = []
    y_wall = []
    for i in range(len(x_robot_corr)):
        sensor_heading = theta_robot[i] + SENSOR_ANGLE_OFFSET
        wx = x_robot_corr[i] + distance_measured[i] * math.cos(sensor_heading)
        wy = y_robot_corr[i] + distance_measured[i] * math.sin(sensor_heading)
        x_wall.append(wx)
        y_wall.append(wy)

    # 3. RANSAC Extraction
    print("Extracting Walls...")
    walls = extract_walls_ransac(x_wall, y_wall)
    
    # --- Correcting Wall Order (GEOMETRIC SORT) ---
    # Step A: Find the centroid (center of mass) of the entire map
    all_wall_x = []
    all_wall_y = []
    for w in walls:
        all_wall_x.extend(w['x_points'])
        all_wall_y.extend(w['y_points'])
    
    center_x = np.mean(all_wall_x)
    center_y = np.mean(all_wall_y)
    
    # Step B: Calculate angle of each wall relative to the center
    for w in walls:
        # Find the center of the specific wall
        wall_mid_x = np.mean(w['x_points'])
        wall_mid_y = np.mean(w['y_points'])
        # Calculate angle (atan2 returns -pi to pi)
        w['angle_to_center'] = math.atan2(wall_mid_y - center_y, wall_mid_x - center_x)
    
    # Step C: Sort walls by angle (ensures circular/clockwise order)
    sorted_walls = sorted(walls, key=lambda x: x['angle_to_center'])
    
    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # Plot original wall points
    plt.plot(x_wall, y_wall, '.', color='lightgray', markersize=2, label='Points')
    
    lines = [] # Store lines in sorted order
    
    # Plot identified walls
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_walls)))
    for i, w in enumerate(sorted_walls):
        m, c = w['m'], w['c']
        lines.append((m, c))
        
        # Plot wall points
        plt.plot(w['x_points'], w['y_points'], '.', color=colors[i], markersize=4)
        
        # Plot fitted line
        wx = w['x_points']
        if m != float('inf'):
            x_range = np.linspace(min(wx), max(wx), 10)
            y_range = m * x_range + c
            plt.plot(x_range, y_range, '-', color=colors[i], linewidth=2)
        else:
            plt.vlines(c, min(w['y_points']), max(w['y_points']), colors[i], linewidth=2)

    # --- Corner Calculation & Labels ---
    corners_x = []
    corners_y = []
    
    if len(lines) >= 3:
        for i in range(len(lines)):
            # Connect wall i to wall i+1 (modulo length to close loop)
            l1 = lines[i]
            l2 = lines[(i + 1) % len(lines)]
            
            res = find_intersection(l1[0], l1[1], l2[0], l2[1])
            if res:
                corners_x.append(res[0])
                corners_y.append(res[1])
        
        # Close the shape on the graph
        if corners_x:
            corners_x.append(corners_x[0])
            corners_y.append(corners_y[0])
            plt.plot(corners_x, corners_y, 'k--', linewidth=2, marker='o', markersize=8, markerfacecolor='yellow', label='Closed Map')
            
            # Add length labels
            for i in range(len(corners_x) - 1):
                p1_x, p1_y = corners_x[i], corners_y[i]
                p2_x, p2_y = corners_x[i+1], corners_y[i+1]
                
                dist = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                mid_x, mid_y = (p1_x + p2_x) / 2, (p1_y + p2_y) / 2
                
                plt.text(mid_x, mid_y, f"{dist:.1f}", fontsize=9, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    plt.title('Final Map: Drift Correction + Geometric Sort', fontsize=16)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# --- Load Data ---
def load_and_process_data():
    try:
        data = np.loadtxt(FILE_NAME, delimiter=',')
        x_robot = data[:, 0] / X_Y_SCALE
        y_robot = data[:, 1] / X_Y_SCALE
        theta_robot = data[:, 2] / THETA_SCALE
        distance_measured = data[:, 3]
        
        process_map_data(x_robot, y_robot, theta_robot, distance_measured)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    load_and_process_data()