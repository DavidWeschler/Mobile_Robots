import numpy as np
import matplotlib.pyplot as plt
import math
import random
from skimage.transform import probabilistic_hough_line

# --- Configuration ---
FILE_NAME = "EX2/plots/MAP.TXT"
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

def get_line_params_from_points(p1, p2):
    """Returns m, c for y = mx + c given two points."""
    x1, y1 = p1
    x2, y2 = p2
    
    if abs(x2 - x1) < 1e-5: # Vertical line
        return float('inf'), x1
        
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c

def find_intersection(m1, c1, m2, c2):
    if m1 == float('inf') and m2 == float('inf'): return None
    if m1 == float('inf'): return c1, m2 * c1 + c2
    if m2 == float('inf'): return c2, m1 * c2 + c1
    if abs(m1 - m2) < 1e-5: return None 
    
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

# --- 3. Hough Transform Logic ---

def points_to_grid(x_pts, y_pts, resolution=2.0):
    """
    Converts float coordinates to a boolean grid (image) for Hough Transform.
    resolution: Size of one grid cell in real-world units.
    """
    if len(x_pts) == 0: return None, None, None, None

    min_x, max_x = min(x_pts), max(x_pts)
    min_y, max_y = min(y_pts), max(y_pts)
    
    # Add padding
    padding = 10.0
    width = int((max_x - min_x + 2*padding) / resolution)
    height = int((max_y - min_y + 2*padding) / resolution)
    
    grid = np.zeros((height, width), dtype=bool)
    
    # Map real coords to grid indices
    img_x_offset = min_x - padding
    img_y_offset = min_y - padding
    
    for x, y in zip(x_pts, y_pts):
        ix = int((x - img_x_offset) / resolution)
        iy = int((y - img_y_offset) / resolution)
        if 0 <= ix < width and 0 <= iy < height:
            grid[iy, ix] = True # Note: y is row, x is col
            
    return grid, img_x_offset, img_y_offset, resolution

def extract_walls_hough(x_all, y_all, params):
    """
    Extracts lines using Probabilistic Hough Transform.
    """
    # 1. Convert points to Image Grid
    res = 1.0 # 1 cm resolution
    grid, off_x, off_y, res = points_to_grid(x_all, y_all, resolution=res)
    
    if grid is None: return []

    # 2. Run Hough
    lines_p = probabilistic_hough_line(grid, 
                                       threshold=params['thresh'], 
                                       line_length=params['len'], 
                                       line_gap=params['gap'])
    
    found_walls = []
    
    # 3. Convert back to world coordinates
    for p0, p1 in lines_p:
        # p0 is (col, row) -> (x, y)
        x0_world = p0[0] * res + off_x
        y0_world = p0[1] * res + off_y
        x1_world = p1[0] * res + off_x
        y1_world = p1[1] * res + off_y
        
        m, c = get_line_params_from_points((x0_world, y0_world), (x1_world, y1_world))
        
        found_walls.append({
            'm': m, 'c': c,
            'x_points': [x0_world, x1_world], 
            'y_points': [y0_world, y1_world]
        })
        
    return found_walls

# --- 4. Processing & Geometry ---

def compute_map_geometry(x_robot_corr, y_robot_corr, theta_robot, dist_meas, hough_params):
    # 1. Calculate Wall Points
    x_wall = []
    y_wall = []
    for i in range(len(x_robot_corr)):
        sensor_heading = theta_robot[i] + SENSOR_ANGLE_OFFSET
        wx = x_robot_corr[i] + dist_meas[i] * math.cos(sensor_heading)
        wy = y_robot_corr[i] + dist_meas[i] * math.sin(sensor_heading)
        x_wall.append(wx)
        y_wall.append(wy)

    # 2. Extract Walls (Hough)
    walls = extract_walls_hough(x_wall, y_wall, hough_params)
    
    if not walls:
        return x_wall, y_wall, [], [], []

    # 3. Geometric Sort (Centroid logic)
    all_pts_x = [pt for w in walls for pt in w['x_points']]
    all_pts_y = [pt for w in walls for pt in w['y_points']]
    center_x = np.mean(all_pts_x) if all_pts_x else 0
    center_y = np.mean(all_pts_y) if all_pts_y else 0
    
    for w in walls:
        wall_mid_x = np.mean(w['x_points'])
        wall_mid_y = np.mean(w['y_points'])
        w['angle_to_center'] = math.atan2(wall_mid_y - center_y, wall_mid_x - center_x)
    
    sorted_walls = sorted(walls, key=lambda x: x['angle_to_center'])
    
    # 4. Find Corners
    corners_x = []
    corners_y = []
    
    lines = [(w['m'], w['c']) for w in sorted_walls]
    
    if len(lines) >= 3:
        for i in range(len(lines)):
            l1 = lines[i]
            l2 = lines[(i + 1) % len(lines)]
            res = find_intersection(l1[0], l1[1], l2[0], l2[1])
            if res:
                corners_x.append(res[0])
                corners_y.append(res[1])
        
        # Close loop
        if corners_x:
            corners_x.append(corners_x[0])
            corners_y.append(corners_y[0])

    return x_wall, y_wall, sorted_walls, corners_x, corners_y

# --- 5. Main Execution ---

def main():
    try:
        # Load Data
        data = np.loadtxt(FILE_NAME, delimiter=',')
        x_robot = data[:, 0] / X_Y_SCALE
        y_robot = data[:, 1] / X_Y_SCALE
        theta_robot = data[:, 2] / THETA_SCALE
        distance_measured = data[:, 3]
        
        # Correct Drift once
        x_corr, y_corr = correct_odometry_drift(x_robot, y_robot)

        # --- Generate 9 Hyperparameter Sets ---
        thresholds = [10, 25, 40]
        line_gaps = [3, 8, 15]
        
        param_sets = []
        for t in thresholds:
            for g in line_gaps:
                param_sets.append({
                    'thresh': t, 
                    'gap': g, 
                    'len': 20
                })
        
        # --- Plotting ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle(f"Hough Transform: 9 Hyperparameter Configurations", fontsize=16)
        axes = axes.flatten()

        for i, params in enumerate(param_sets):
            ax = axes[i]
            
            xw, yw, walls, cx, cy = compute_map_geometry(x_corr, y_corr, theta_robot, distance_measured, params)
            
            # Plot Raw Points
            ax.plot(xw, yw, '.', color='lightgray', markersize=1)
            
            # Plot Walls
            colors = plt.cm.jet(np.linspace(0, 1, len(walls)))
            for j, w in enumerate(walls):
                ax.plot(w['x_points'], w['y_points'], '-', color=colors[j], linewidth=2, alpha=0.7)
            
            # Plot Corners and Length Labels
            if cx:
                ax.plot(cx, cy, 'k--', marker='o', markerfacecolor='yellow', markersize=5, linewidth=1.5)
                
                # --- ADDED: Length Labels ---
                for k in range(len(cx) - 1):
                    p1_x, p1_y = cx[k], cy[k]
                    p2_x, p2_y = cx[k+1], cy[k+1]
                    
                    dist = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                    mid_x, mid_y = (p1_x + p2_x) / 2, (p1_y + p2_y) / 2
                    
                    ax.text(mid_x, mid_y, f"{dist:.1f}", fontsize=8, fontweight='bold',
                            color='black', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))

            ax.set_title(f"Thresh:{params['thresh']} | Gap:{params['gap']} | Len:{params['len']}")
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()