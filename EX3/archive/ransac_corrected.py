import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = r"C:\CS\Robots_mobile\Mobile_Robots\EX3\GUY_TRACK_LOG.TXT"
X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) 

# Fix randomness
random.seed(42)
np.random.seed(42)

# --- 1. Drift Correction ---

def correct_odometry_drift(x_robot, y_robot):
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

# --- 2. Math Helpers ---

def rotate_points(x_arr, y_arr, angle_rad):
    """ Rotates arrays of X and Y by a specific angle (radians) """
    x_new = []
    y_new = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    for x, y in zip(x_arr, y_arr):
        nx = x * cos_a - y * sin_a
        ny = x * sin_a + y * cos_a
        x_new.append(nx)
        y_new.append(ny)
    return x_new, y_new

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
    if len(x) == 0: return 0, 0
    if np.std(x) < 0.01 * np.std(y): return float('inf'), np.mean(x)
    
    A = np.hstack([x, np.ones_like(x)])
    try:
        A_pinv = np.linalg.pinv(A)
        p = A_pinv @ y
        return p[0][0], p[1][0]
    except:
        return 0, 0

def find_intersection(m1, c1, m2, c2):
    if m1 == float('inf') and m2 == float('inf'): return None
    if m1 == float('inf'): return c1, m2 * c1 + c2
    if m2 == float('inf'): return c2, m1 * c2 + c1
    if abs(m1 - m2) < 1e-5: return None 
    
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

def extract_walls_ransac(x_all, y_all, max_distance=5.0, min_points=7, iterations=1000):
    remaining_indices = list(range(len(x_all)))
    found_walls = []
    
    while len(remaining_indices) > min_points:
        best_inliers_indices = []
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
            
            current_inliers = []
            for i, d in enumerate(dists):
                if d < max_distance:
                    current_inliers.append(remaining_indices[i])
            
            if len(current_inliers) > len(best_inliers_indices):
                best_inliers_indices = current_inliers
        
        wall_x = [x_all[i] for i in best_inliers_indices]
        wall_y = [y_all[i] for i in best_inliers_indices]
        m, c = fit_line_pseudo_inverse(wall_x, wall_y)  # (LSQ) Least Squares fit on inliers
        
        found_walls.append({
            'm': m, 'c': c,
            'x_points': wall_x, 
            'y_points': wall_y,
            'indices': best_inliers_indices 
        })
        
        remaining_indices = [idx for idx in remaining_indices if idx not in best_inliers_indices]
        
    return found_walls

# --- 3. Main Logic ---

def calculate_and_save_center(corners_x, corners_y, sorted_walls, start_x, start_y, total_rotation):
    unique_corners_x = []
    unique_corners_y = []

    if corners_x:
        # Use centroid of corners (polygon)
        # corners_x has the first point repeated at the end, so exclude it
        unique_corners_x = corners_x[:-1]
        unique_corners_y = corners_y[:-1]
    
    # Prepare list of corners relative to start (0,0) in robot's reference frame
    # Robot starts at origin facing NEGATIVE X axis, so we:
    # 1. Compute offset from start position (both in aligned/rotated frame)
    # 2. Rotate by 180° (negate both) since robot faces -X, not +X
    final_corners = []
    
    if unique_corners_x:
        for cx, cy in zip(unique_corners_x, unique_corners_y):
            # Vector from start (in aligned frame)
            dx = cx - start_x
            dy = cy - start_y
            
            # Robot faces negative X, so rotate 180° to robot's frame
            # Rotation by π: (x, y) -> (-x, -y)
            final_corners.append((-dx, -dy))
    
    # Write to file
    try:
        with open("EX2/plots/CENTER.TXT", "w") as f:
            f.write(f"{len(final_corners)}\n")
            for x, y in final_corners:
                f.write(f"{x}\n")
                f.write(f"{y}\n")
        print(f"Successfully wrote EX2/plots/CENTER.TXT with {len(final_corners)} corners")
    except Exception as e:
        print(f"Failed to write CENTER.TXT: {e}")

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
    
    # 4. Geometric Sort
    all_wall_x = []
    all_wall_y = []
    for w in walls:
        all_wall_x.extend(w['x_points'])
        all_wall_y.extend(w['y_points'])
    
    center_x = np.mean(all_wall_x)
    center_y = np.mean(all_wall_y)
    
    for w in walls:
        wall_mid_x = np.mean(w['x_points'])
        wall_mid_y = np.mean(w['y_points'])
        w['angle_to_center'] = math.atan2(wall_mid_y - center_y, wall_mid_x - center_x)
    
    sorted_walls = sorted(walls, key=lambda x: x['angle_to_center'])
    
    # --- 5. TEMPORAL ALIGNMENT & ORIENTATION CHECK ---
    
    # Track start point and rotation
    start_x = x_robot_corr[0]
    start_y = y_robot_corr[0]
    total_rotation = 0.0

    anchor_wall = None
    min_index_found = float('inf')
    
    for w in sorted_walls:
        if not w['indices']: continue
        earliest_idx_in_wall = min(w['indices'])
        if earliest_idx_in_wall < min_index_found:
            min_index_found = earliest_idx_in_wall
            anchor_wall = w
            
    if anchor_wall:
        print(f"Anchor wall found (Start Index: {min_index_found})")
        m_anchor = anchor_wall['m']
        
        # Calculate initial rotation to make line horizontal
        if m_anchor == float('inf'):
            current_angle = math.pi / 2
        else:
            current_angle = math.atan(m_anchor)
            
        rotation_angle = -current_angle
        total_rotation += rotation_angle
        
        # Rotate start point
        sx, sy = rotate_points([start_x], [start_y], rotation_angle)
        start_x, start_y = sx[0], sy[0]
        
        # 5a. Apply First Rotation (Horizontal Alignment)
        all_x_temp = []
        all_y_temp = []
        
        for w in sorted_walls:
            rx, ry = rotate_points(w['x_points'], w['y_points'], rotation_angle)
            w['x_points'] = rx
            w['y_points'] = ry
            all_x_temp.extend(rx)
            all_y_temp.extend(ry)
            
        # 5b. Orientation Check (Upside Down Check)
        # Calculate the Y-centroid of the whole map vs the Anchor Wall
        map_centroid_y = np.mean(all_y_temp)
        anchor_centroid_y = np.mean(anchor_wall['y_points'])
        
        # If the map centroid is BELOW the anchor wall, we are upside down.
        # We want the room to be "above" (positive Y) the starting wall.
        if map_centroid_y < anchor_centroid_y:
            print("Map is upside down. Flipping 180 degrees...")
            rotation_fix = math.pi # 180 degrees
            total_rotation += rotation_fix
            
            # Rotate start point
            sx, sy = rotate_points([start_x], [start_y], rotation_fix)
            start_x, start_y = sx[0], sy[0]
            
            for w in sorted_walls:
                rx, ry = rotate_points(w['x_points'], w['y_points'], rotation_fix)
                w['x_points'] = rx
                w['y_points'] = ry
        
        # 5c. Re-fit lines after all rotations
        for w in sorted_walls:
            nm, nc = fit_line_pseudo_inverse(w['x_points'], w['y_points'])  # (LSQ) Least Squares fit on inliers
            w['m'] = nm
            w['c'] = nc

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    lines = [] 
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_walls)))
    
    for i, w in enumerate(sorted_walls):
        m, c = w['m'], w['c']
        lines.append((m, c))
        
        plt.plot(w['x_points'], w['y_points'], '.', color=colors[i], markersize=4)
        
        wx = w['x_points']
        if not wx: continue
        
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
        
        if corners_x:
            corners_x.append(corners_x[0])
            corners_y.append(corners_y[0])
            plt.plot(corners_x, corners_y, 'k--', linewidth=2, marker='o', markersize=8, markerfacecolor='yellow', label='Closed Map')
            
            for i in range(len(corners_x) - 1):
                p1_x, p1_y = corners_x[i], corners_y[i]
                p2_x, p2_y = corners_x[i+1], corners_y[i+1]
                
                dist = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                mid_x, mid_y = (p1_x + p2_x) / 2, (p1_y + p2_y) / 2
                
                plt.text(mid_x, mid_y, f"{dist:.1f}", fontsize=9, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    # --- Calculate Center and Write Output ---
    calculate_and_save_center(corners_x, corners_y, sorted_walls, start_x, start_y, total_rotation)

    plt.title('Final Map: Aligned (Room Above Start)', fontsize=16)
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