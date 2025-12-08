import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = "EX2/plots/MAP.TXT"
X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) 

# Hough Parameters
RHO_RESOLUTION = 2.0    # cm per bin
THETA_RESOLUTION = 1.0  # degrees per bin
MIN_VOTES = 25          # Minimum points to consider a line
MAX_WALLS = 8          # Safety limit to stop infinite loops
DIST_THRESHOLD = 6.0    # cm - Distance to line to be considered an inlier

# Fix randomness
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
        factor = i / (num_points - 1)
        x_corrected[i] -= drift_x * factor
        y_corrected[i] -= drift_y * factor
        
    return x_corrected, y_corrected

# --- 2. Math & Hough Helpers ---

def find_intersection(m1, c1, m2, c2):
    """Finds intersection (x, y) of two lines y=mx+c."""
    if m1 == float('inf') and m2 == float('inf'): return None
    if m1 == float('inf'): return c1, m2 * c1 + c2
    if m2 == float('inf'): return c2, m1 * c2 + c1
    if abs(m1 - m2) < 1e-5: return None 
    
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

def fit_line_least_squares(x_points, y_points):
    """
    Fits a line to a set of points using Linear Least Squares (Pseudo-Inverse).
    Returns (slope m, intercept c). Handles vertical lines.
    """
    x = np.array(x_points)
    y = np.array(y_points)
    
    # Check if vertical (variance in x is tiny compared to y)
    if np.std(x) < 0.01 * np.std(y): 
        return float('inf'), np.mean(x)

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def perform_hough_transform(x, y, rho_res=1.0, theta_step=1.0):
    """
    Vectorized Hough Transform.
    Returns: accumulator (grid), thetas (radians), rhos (values)
    """
    # 1. Define Theta Range (-90 to 90 degrees)
    thetas_deg = np.arange(-90, 90, theta_step)
    thetas_rad = np.deg2rad(thetas_deg)
    
    # 2. Precompute Sin/Cos
    cos_t = np.cos(thetas_rad)
    sin_t = np.sin(thetas_rad)
    
    # 3. Calculate Rho for every point and every angle
    # Shape: (num_points, num_thetas)
    # rho = x * cos(theta) + y * sin(theta)
    rhos_calc = np.outer(x, cos_t) + np.outer(y, sin_t)
    
    # 4. Binning
    # Find global min/max rho to define the grid size
    max_rho_val = np.ceil(np.max(np.abs(rhos_calc)))
    # Create bins from -max to +max
    rho_bins = np.arange(-max_rho_val, max_rho_val, rho_res)
    
    # 5. Accumulate votes
    accumulator = np.zeros((len(rho_bins), len(thetas_rad)), dtype=np.int32)
    
    # Digitizing maps the calculated rho values to bin indices
    for i in range(len(thetas_rad)):
        # Get rhos for this specific angle across all points
        col_rhos = rhos_calc[:, i]
        # Find which bin they fall into
        bin_indices = np.digitize(col_rhos, rho_bins) - 1
        
        # Filter out of bounds (just safety)
        valid_mask = (bin_indices >= 0) & (bin_indices < len(rho_bins))
        valid_indices = bin_indices[valid_mask]
        
        # Count occurrences (votes)
        unique_bins, counts = np.unique(valid_indices, return_counts=True)
        accumulator[unique_bins, i] = counts

    return accumulator, thetas_rad, rho_bins

def extract_walls_hough(x_all, y_all, dist_threshold=DIST_THRESHOLD):
    """
    Iterative Hough Transform:
    1. Run Hough.
    2. Find best line.
    3. Remove inliers.
    4. Repeat.
    """
    remaining_x = np.array(x_all)
    remaining_y = np.array(y_all)
    found_walls = []
    
    iteration = 0
    
    while len(remaining_x) > MIN_VOTES and iteration < MAX_WALLS:
        print(f"  Hough Iteration {iteration+1} (Points left: {len(remaining_x)})")
        
        # 1. Run Hough on remaining points
        acc, thetas, rhos = perform_hough_transform(
            remaining_x, remaining_y, 
            rho_res=RHO_RESOLUTION, 
            theta_step=THETA_RESOLUTION
        )
        
        # 2. Find Peak (Highest Vote)
        if np.max(acc) < MIN_VOTES:
            break
            
        # Get index of max vote
        idx = np.unravel_index(np.argmax(acc), acc.shape)
        rho_idx, theta_idx = idx
        
        best_rho = rhos[rho_idx]
        best_theta = thetas[theta_idx]
        
        # 3. Find Inliers (Geometric distance from point to line)
        # Line eq: x*cos(t) + y*sin(t) - rho = 0
        distances = np.abs(remaining_x * np.cos(best_theta) + 
                           remaining_y * np.sin(best_theta) - best_rho)
        
        inlier_mask = distances < dist_threshold
        
        # 4. Extract Inlier Points
        wall_x = remaining_x[inlier_mask]
        wall_y = remaining_y[inlier_mask]
        
        # If we accidentally grabbed too few points due to discretization error
        if len(wall_x) < MIN_VOTES:
            # Zero out this peak and try again without re-calculating everything
            acc[rho_idx, theta_idx] = 0
            continue

        # 5. Re-Fit Line using Least Squares (Better accuracy than Hough bin)
        m, c = fit_line_least_squares(wall_x, wall_y)
        
        found_walls.append({
            'm': m, 'c': c,
            'x_points': wall_x, 'y_points': wall_y,
            'rho': best_rho, 'theta': best_theta # Store raw hough params too just in case
        })
        
        # 6. Remove inliers from dataset
        remaining_x = remaining_x[~inlier_mask]
        remaining_y = remaining_y[~inlier_mask]
        
        iteration += 1
        
    return found_walls

# --- 3. Main Logic ---

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

    # 3. Hough Extraction
    print("Extracting Walls (Iterative Hough)...")
    walls = extract_walls_hough(x_wall, y_wall)
    
    # --- Correcting Wall Order (GEOMETRIC SORT) ---
    # Step A: Find the centroid of all detected wall points
    all_wall_x = []
    all_wall_y = []
    for w in walls:
        all_wall_x.extend(w['x_points'])
        all_wall_y.extend(w['y_points'])
    
    if not all_wall_x:
        print("No walls found!")
        return

    center_x = np.mean(all_wall_x)
    center_y = np.mean(all_wall_y)
    
    # Step B: Calculate angle of each wall relative to the center
    for w in walls:
        wall_mid_x = np.mean(w['x_points'])
        wall_mid_y = np.mean(w['y_points'])
        w['angle_to_center'] = math.atan2(wall_mid_y - center_y, wall_mid_x - center_x)
    
    # Step C: Sort walls by angle
    sorted_walls = sorted(walls, key=lambda x: x['angle_to_center'])
    
    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # Plot original points
    plt.plot(x_wall, y_wall, '.', color='lightgray', markersize=2, label='Points')
    
    lines = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_walls)))
    
    for i, w in enumerate(sorted_walls):
        m, c = w['m'], w['c']
        lines.append((m, c))
        
        # Plot points belonging to this wall
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
            l1 = lines[i]
            l2 = lines[(i + 1) % len(lines)]
            
            res = find_intersection(l1[0], l1[1], l2[0], l2[1])
            if res:
                corners_x.append(res[0])
                corners_y.append(res[1])
        
        # Close the shape
        if corners_x:
            corners_x.append(corners_x[0])
            corners_y.append(corners_y[0])
            plt.plot(corners_x, corners_y, 'k--', linewidth=1.5, marker='o', 
                     markersize=8, markerfacecolor='yellow', label='Closed Map')
            
            # Add length labels
            for i in range(len(corners_x) - 1):
                p1_x, p1_y = corners_x[i], corners_y[i]
                p2_x, p2_y = corners_x[i+1], corners_y[i+1]
                
                dist = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                mid_x, mid_y = (p1_x + p2_x) / 2, (p1_y + p2_y) / 2
                
                plt.text(mid_x, mid_y, f"{dist:.1f}", fontsize=9, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    plt.title('Final Map: Drift Correction + Iterative Hough', fontsize=16)
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