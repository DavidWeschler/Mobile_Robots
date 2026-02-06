import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\SLAM_CLEANED.TXT"

# --- TUNED PARAMETERS FOR RANSAC ---
RANSAC_ITERATIONS = 100      # Number of RANSAC iterations per wall
RANSAC_THRESHOLD = 8.0       # Distance threshold for inliers (cm)
RANSAC_MIN_INLIERS = 6       # Minimum points to form a wall
MIN_POINTS_REMAINING = 6     # Stop when fewer points remain

# Fix randomness
random.seed(42)
np.random.seed(42)

# ==============================================================================
# 1. RANSAC LINE DETECTION
# ==============================================================================

def fit_line_least_squares(x_pts, y_pts):
    if len(x_pts) < 2: return 0, 0
    x = np.array(x_pts)
    y = np.array(y_pts)
    if np.std(y) > np.std(x) * 3:  # Vertical line
        return float('inf'), np.mean(x)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def point_to_line_distance(x, y, m, c):
    """Calculate perpendicular distance from point to line y = mx + c"""
    if m == float('inf'):
        return abs(x - c)
    # Line: mx - y + c = 0, distance = |mx - y + c| / sqrt(m^2 + 1)
    return abs(m * x - y + c) / math.sqrt(m**2 + 1)

def ransac_fit_line(x_pts, y_pts):
    """
    RANSAC to fit a single line to points.
    Returns (m, c, inlier_indices) or None if no good line found.
    """
    n_points = len(x_pts)
    if n_points < 2:
        return None
    
    best_inliers = []
    best_m, best_c = 0, 0
    
    for _ in range(RANSAC_ITERATIONS):
        # 1. Randomly select 2 points
        idx1, idx2 = random.sample(range(n_points), 2)
        x1, y1 = x_pts[idx1], y_pts[idx1]
        x2, y2 = x_pts[idx2], y_pts[idx2]
        
        # 2. Fit line through these 2 points
        if abs(x2 - x1) < 1e-6:  # Vertical line
            m = float('inf')
            c = (x1 + x2) / 2
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
        
        # 3. Count inliers
        inliers = []
        for i in range(n_points):
            dist = point_to_line_distance(x_pts[i], y_pts[i], m, c)
            if dist < RANSAC_THRESHOLD:
                inliers.append(i)
        
        # 4. Keep best
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m, best_c = m, c
    
    if len(best_inliers) < RANSAC_MIN_INLIERS:
        return None
    
    # 5. Refit line using all inliers
    inlier_x = [x_pts[i] for i in best_inliers]
    inlier_y = [y_pts[i] for i in best_inliers]
    final_m, final_c = fit_line_least_squares(inlier_x, inlier_y)
    
    return final_m, final_c, best_inliers

def extract_walls_ransac(x_all, y_all):
    """
    Extract multiple walls using iterative RANSAC.
    """
    remaining_idx = list(range(len(x_all)))
    found_walls = []
    
    print(f"  > Running RANSAC on {len(x_all)} points...")
    
    while len(remaining_idx) >= MIN_POINTS_REMAINING:
        # Get current remaining points
        curr_x = [x_all[i] for i in remaining_idx]
        curr_y = [y_all[i] for i in remaining_idx]
        
        # Try to fit a line
        result = ransac_fit_line(curr_x, curr_y)
        
        if result is None:
            break
        
        m, c, local_inliers = result
        
        # Convert local indices to global indices
        real_indices = [remaining_idx[k] for k in local_inliers]
        
        # Store wall data
        wall_x = [x_all[i] for i in real_indices]
        wall_y = [y_all[i] for i in real_indices]
        
        found_walls.append({
            'm': m, 'c': c,
            'x_points': wall_x, 'y_points': wall_y, 'indices': real_indices
        })
        
        print(f"    -> Found Wall with {len(real_indices)} points.")
        
        # Remove inliers from remaining points
        remaining_idx = [idx for idx in remaining_idx if idx not in real_indices]
    
    return found_walls

# ==============================================================================
# 2. HELPERS & SPRING CORRECTION
# ==============================================================================

def rotate_points(x_arr, y_arr, angle_rad):
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

def find_intersection(m1, c1, m2, c2):
    if m1 == float('inf') and m2 == float('inf'): return None
    if m1 == float('inf'): return c1, m2 * c1 + c2
    if m2 == float('inf'): return c2, m1 * c2 + c1
    if abs(m1 - m2) < 1e-5: return None 
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

def apply_spring_correction(rx, ry, wx, wy):
    """
    Applies the Spring Method to close the loop.
    Assumes index 0 and index -1 of the robot path should be identical.
    Distributes the error linearly across the path.
    """
    n = len(rx)
    if n < 2: return rx, ry, wx, wy

    # 1. Calculate the Error (Drift)
    # The discrepancy between the last point and the first point
    error_x = rx[-1] - rx[0]
    error_y = ry[-1] - ry[0]

    print(f"  > Loop Closure Error detected: X={error_x:.2f}cm, Y={error_y:.2f}cm")
    print("  > Applying Spring Correction...")

    # 2. Apply Correction Vector linearly
    rx_fixed = np.copy(rx)
    ry_fixed = np.copy(ry)
    wx_fixed = np.copy(wx)
    wy_fixed = np.copy(wy)

    for i in range(n):
        # Linear weight: 0 at start, 1 at end
        alpha = i / (n - 1)
        
        correction_x = error_x * alpha
        correction_y = error_y * alpha

        # Correct Robot Position
        rx_fixed[i] -= correction_x
        ry_fixed[i] -= correction_y

        # Correct Wall Point
        # Since the wall point is relative to the robot's sensor frame, 
        # shifting the robot frame shifts the wall point by the same vector.
        wx_fixed[i] -= correction_x
        wy_fixed[i] -= correction_y

    return rx_fixed, ry_fixed, wx_fixed, wy_fixed

# ==============================================================================
# 3. INTERACTIVE VISUALIZER
# ==============================================================================

class InteractiveMapVisualizer:
    def __init__(self, sorted_walls, corners_x, corners_y, raw_x, raw_y):
        self.walls = sorted_walls
        self.corners_x = corners_x
        self.corners_y = corners_y
        self.raw_x = raw_x
        self.raw_y = raw_y
        
        # State for transformations
        self.mirror_factor = 1  # 1 for normal, -1 for mirrored
        self.rotation_angle = 0 # Radians
        
        # Setup Figure with 2 Subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial Plot
        self.plot_ideal()
        self.update_map_plot()
        
        plt.show()

    def on_key(self, event):
        if event.key == 'r':
            self.rotation_angle += math.pi / 2
            print(" -> Rotated 90 degrees")
            self.update_map_plot()
        elif event.key == 'm':
            self.mirror_factor *= -1
            print(" -> Mirrored Map")
            self.update_map_plot()

    def transform_points(self, x_list, y_list):
        # 1. Mirror (flip X)
        x_m = [x * self.mirror_factor for x in x_list]
        y_m = y_list
        # 2. Rotate
        if self.rotation_angle != 0:
            return rotate_points(x_m, y_m, self.rotation_angle)
        return x_m, y_m

    def plot_ideal(self):
        self.ax1.set_title("Ideal Arena", fontsize=14)
        ARENA_PTS = [(-41.5, -30), (-130.5, 59), (-130.5, 207), (59.5, 207), (59.5, -30)]
        real_x = [p[0] for p in ARENA_PTS]
        real_y = [p[1] for p in ARENA_PTS]
        
        # Close the loop for plotting
        real_x_plot = real_x + [real_x[0]]
        real_y_plot = real_y + [real_y[0]]
        
        self.ax1.plot(real_x_plot, real_y_plot, 'k-', linewidth=2, label="Ideal Arena")
        self.ax1.fill(real_x_plot, real_y_plot, alpha=0.1, color='gray')
        
        # Annotate
        for i in range(len(real_x)):
            p1 = (real_x[i], real_y[i])
            p2 = (real_x[(i+1)%len(real_x)], real_y[(i+1)%len(real_y)])
            dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            self.ax1.text(mid[0], mid[1], f"{dist:.1f}", fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            
        self.ax1.axis('equal')
        self.ax1.grid(True)

    def update_map_plot(self):
        self.ax2.clear()
        self.ax2.set_title("Generated Map - Spring Corrected + RANSAC\n(Controls: 'r' to rotate, 'm' to mirror)", fontsize=14)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.walls)))

        # Plot Walls
        for i, w in enumerate(self.walls):
            if not w['x_points']: continue
            
            # Transform raw points
            tx, ty = self.transform_points(w['x_points'], w['y_points'])
            
            # Plot points
            self.ax2.plot(tx, ty, '.', color=colors[i], markersize=4)
            
            # Fit line for visualization on transformed points
            nm, nc = fit_line_least_squares(tx, ty)
            
            # Draw line segment
            if nm != float('inf'):
                x_range = np.linspace(min(tx), max(tx), 10)
                y_range = nm * x_range + nc
                self.ax2.plot(x_range, y_range, '-', color=colors[i], linewidth=2)
            else:
                self.ax2.vlines(nc, min(ty), max(ty), colors[i], linewidth=2)

        # Plot Corners
        if self.corners_x:
            cx_trans, cy_trans = self.transform_points(self.corners_x, self.corners_y)
            self.ax2.plot(cx_trans, cy_trans, 'k--', linewidth=2, marker='o', markersize=8, markerfacecolor='yellow', label='Map')
            
            # Distance annotations
            for i in range(len(cx_trans) - 1):
                p1_x, p1_y = cx_trans[i], cy_trans[i]
                p2_x, p2_y = cx_trans[i+1], cy_trans[i+1]
                dist = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                mid_x, mid_y = (p1_x + p2_x) / 2, (p1_y + p2_y) / 2
                self.ax2.text(mid_x, mid_y, f"{dist:.1f}", fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

        self.ax2.axis('equal')
        self.ax2.grid(True)
        self.fig.canvas.draw()

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================

def process_map_data(x_wall, y_wall):
    
    print(f"1. Total points for wall detection: {len(x_wall)}")

    print("2. Extracting Walls (RANSAC)...")
    walls = extract_walls_ransac(x_wall, y_wall)
    
    if not walls:
        print("NO WALLS FOUND! Plotting raw points for debug...")
        plt.figure()
        plt.scatter(x_wall, y_wall, s=5)
        plt.title("Debug: Raw Wall Points (No Lines Detected)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        return

    # 3. Sort Walls Geometrically
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
    
    # 4. Alignment - align the first wall to be horizontal/vertical
    anchor_wall = None
    min_index_found = float('inf')
    
    for w in sorted_walls:
        if not w['indices']: continue
        earliest_idx_in_wall = min(w['indices'])
        if earliest_idx_in_wall < min_index_found:
            min_index_found = earliest_idx_in_wall
            anchor_wall = w
            
    if anchor_wall:
        m_anchor = anchor_wall['m']
        if m_anchor == float('inf'): 
            current_angle = math.pi / 2
        else: 
            current_angle = math.atan(m_anchor)
        
        rotation_angle = -current_angle
        
        all_y_temp = []
        for w in sorted_walls:
            rx, ry = rotate_points(w['x_points'], w['y_points'], rotation_angle)
            w['x_points'] = rx
            w['y_points'] = ry
            all_y_temp.extend(ry)
            
        anchor_centroid_y = np.mean(anchor_wall['y_points'])
        map_centroid_y = np.mean(all_y_temp)
        
        if map_centroid_y < anchor_centroid_y:
            rotation_fix = math.pi
            for w in sorted_walls:
                rx, ry = rotate_points(w['x_points'], w['y_points'], rotation_fix)
                w['x_points'] = rx
                w['y_points'] = ry

        for w in sorted_walls:
            nm, nc = fit_line_least_squares(w['x_points'], w['y_points'])
            w['m'] = nm
            w['c'] = nc

    # --- CALCULATE CORNERS ---
    corners_x = []
    corners_y = []
    
    # Re-calculate lines based on aligned data
    lines = []
    for w in sorted_walls:
        lines.append((w['m'], w['c']))

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

    # --- INTERACTIVE PLOTTING ---
    InteractiveMapVisualizer(sorted_walls, corners_x, corners_y, x_wall, y_wall)

# --- Load Data ---
def load_and_process_data():
    try:
        print(f"Loading data from: {FILE_NAME}")
        data = np.loadtxt(FILE_NAME, delimiter=',')
        
        # New format: 4 columns [x_robot, y_robot, x_wall, y_wall]
        x_robot = data[:, 0]
        y_robot = data[:, 1]
        x_wall  = data[:, 2]
        y_wall  = data[:, 3]
        
        print(f"Loaded {len(x_robot)} points.")
        
        # --- APPLY SPRING CORRECTION ---
        # Fixes drift by forcing the last robot point to match the first robot point
        xr_fixed, yr_fixed, xw_fixed, yw_fixed = apply_spring_correction(x_robot, y_robot, x_wall, y_wall)
        
        # Process the map with the CORRECTED wall data
        process_map_data(list(xw_fixed), list(yw_fixed))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_and_process_data()