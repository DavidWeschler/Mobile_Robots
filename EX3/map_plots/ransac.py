import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\map_plots\\SLAM_PATH.TXT"
X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) 

# Fix randomness
random.seed(42)
np.random.seed(42)

# ==============================================================================
# 1. SPRING METHOD (GRAPH SLAM) REPLACES LINEAR DRIFT CORRECTION
# ==============================================================================

class GraphSLAM:
    def __init__(self):
        self.nodes = []       # Robot Poses
        self.landmarks = []   # Wall Points (Measurements)
        self.constraints = [] # Springs

    def add_node(self, x, y, theta):
        # First node is fixed (anchor)
        fixed = (len(self.nodes) == 0)
        self.nodes.append({'x': x, 'y': y, 'theta': theta, 'fixed': fixed})

        # Add Odometry Constraint (connect to previous node)
        if len(self.nodes) > 1:
            prev_idx = len(self.nodes) - 2
            curr_idx = len(self.nodes) - 1
            prev = self.nodes[prev_idx]
            
            self.constraints.append({
                'type': 'odo',
                'i': prev_idx, 
                'j': curr_idx,
                'dx': x - prev['x'],
                'dy': y - prev['y']
            })

    def add_measurement(self, node_idx, dist):
        robot = self.nodes[node_idx]
        angle = robot['theta'] + SENSOR_ANGLE_OFFSET
        lx = robot['x'] + dist * math.cos(angle)
        ly = robot['y'] + dist * math.sin(angle)

        # Simple Data Association: Is this point close to an existing landmark?
        best_idx = -1
        min_dist = 20.0 # Threshold in CM (tune this if needed)

        for i, lm in enumerate(self.landmarks):
            d = math.sqrt((lx - lm['x'])**2 + (ly - lm['y'])**2)
            if d < min_dist:
                min_dist = d
                best_idx = i
        
        if best_idx != -1:
            # Merge with existing
            lm = self.landmarks[best_idx]
            lm['x'] = (lm['x'] * lm['n'] + lx) / (lm['n'] + 1)
            lm['y'] = (lm['y'] * lm['n'] + ly) / (lm['n'] + 1)
            lm['n'] += 1
        else:
            # Create new
            self.landmarks.append({'x': lx, 'y': ly, 'n': 1})
            best_idx = len(self.landmarks) - 1

        self.constraints.append({
            'type': 'meas',
            'node_idx': node_idx,
            'land_idx': best_idx,
            'dist': dist,
            'angle_offset': SENSOR_ANGLE_OFFSET
        })

    def relax_springs(self, iterations=50, learning_rate=0.1):
        print(f"  > Relaxing Springs ({iterations} iterations)...")
        
        for _ in range(iterations):
            node_corrections = {i: {'dx': 0, 'dy': 0, 'c': 0} for i in range(len(self.nodes))}
            land_corrections = {i: {'dx': 0, 'dy': 0, 'c': 0} for i in range(len(self.landmarks))}

            for c in self.constraints:
                if c['type'] == 'odo':
                    n1 = self.nodes[c['i']]
                    n2 = self.nodes[c['j']]
                    
                    # Expected position of n2
                    pred_x = n1['x'] + c['dx']
                    pred_y = n1['y'] + c['dy']
                    
                    ex = pred_x - n2['x']
                    ey = pred_y - n2['y']
                    
                    if not n1['fixed']:
                        node_corrections[c['i']]['dx'] += ex * 0.5
                        node_corrections[c['i']]['dy'] += ey * 0.5
                        node_corrections[c['i']]['c'] += 1
                    
                    if not n2['fixed']:
                        node_corrections[c['j']]['dx'] -= ex * 0.5
                        node_corrections[c['j']]['dy'] -= ey * 0.5
                        node_corrections[c['j']]['c'] += 1

                elif c['type'] == 'meas':
                    robot = self.nodes[c['node_idx']]
                    land = self.landmarks[c['land_idx']]
                    
                    angle = robot['theta'] + c['angle_offset']
                    pred_lx = robot['x'] + c['dist'] * math.cos(angle)
                    pred_ly = robot['y'] + c['dist'] * math.sin(angle)
                    
                    ex = pred_lx - land['x']
                    ey = pred_ly - land['y']
                    
                    if not robot['fixed']:
                        node_corrections[c['node_idx']]['dx'] -= ex * 0.5
                        node_corrections[c['node_idx']]['dy'] -= ey * 0.5
                        node_corrections[c['node_idx']]['c'] += 1
                        
                    land_corrections[c['land_idx']]['dx'] += ex * 0.5
                    land_corrections[c['land_idx']]['dy'] += ey * 0.5
                    land_corrections[c['land_idx']]['c'] += 1

            # Apply
            for i, corr in node_corrections.items():
                if corr['c'] > 0 and not self.nodes[i]['fixed']:
                    self.nodes[i]['x'] += (corr['dx'] / corr['c']) * learning_rate
                    self.nodes[i]['y'] += (corr['dy'] / corr['c']) * learning_rate

            for i, corr in land_corrections.items():
                if corr['c'] > 0:
                    self.landmarks[i]['x'] += (corr['dx'] / corr['c']) * learning_rate
                    self.landmarks[i]['y'] += (corr['dy'] / corr['c']) * learning_rate

# ==============================================================================
# 2. MATH HELPERS
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

def get_line_params_from_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) < 1e-6: # Vertical line protection
        return 1.0, 0.0, -x1
    
    m = (y2 - y1) / (x2 - x1)
    # Ax + By + C = 0  => mx - y + (y1 - mx1) = 0
    return m, -1.0, (y1 - m * x1)

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
    
    print(f"  > RANSAC on {len(x_all)} points...")
    
    while len(remaining_indices) > min_points:
        best_inliers_indices = []
        for _ in range(iterations):
            if len(remaining_indices) < 2: break
            sample_idx = random.sample(remaining_indices, 2)
            p1 = (x_all[sample_idx[0]], y_all[sample_idx[0]])
            p2 = (x_all[sample_idx[1]], y_all[sample_idx[1]])
            
            A, B, C = get_line_params_from_2_points(p1, p2)
            
            curr_x = [x_all[i] for i in remaining_indices]
            curr_y = [y_all[i] for i in remaining_indices]
            dists = get_point_line_distances(curr_x, curr_y, A, B, C)
            
            # Find indices where dist < max_distance
            current_inliers = []
            for i, d in enumerate(dists):
                if d < max_distance:
                    current_inliers.append(remaining_indices[i])
            
            if len(current_inliers) > len(best_inliers_indices):
                best_inliers_indices = current_inliers
        
        # If we didn't find enough points, stop
        if len(best_inliers_indices) < min_points:
            break

        wall_x = [x_all[i] for i in best_inliers_indices]
        wall_y = [y_all[i] for i in best_inliers_indices]
        m, c = fit_line_pseudo_inverse(wall_x, wall_y)
        
        found_walls.append({
            'm': m, 'c': c,
            'x_points': wall_x, 
            'y_points': wall_y,
            'indices': best_inliers_indices 
        })
        
        remaining_indices = [idx for idx in remaining_indices if idx not in best_inliers_indices]
        
    return found_walls

# ==============================================================================
# 3. MAIN LOGIC (UPDATED TO USE GRAPH SLAM)
# ==============================================================================

def calculate_and_save_center(corners_x, corners_y, sorted_walls, start_x, start_y, total_rotation):
    unique_corners_x = []
    unique_corners_y = []

    if corners_x:
        unique_corners_x = corners_x[:-1]
        unique_corners_y = corners_y[:-1]
    
    final_corners = []
    
    if unique_corners_x:
        for cx, cy in zip(unique_corners_x, unique_corners_y):
            dx = cx - start_x
            dy = cy - start_y
            # Robot faces negative X, so rotate 180 (negate)
            final_corners.append((-dx, -dy))
    
    try:
        with open("EX2/plots/CENTER.TXT", "w") as f:
            f.write(f"{len(final_corners)}\n")
            for x, y in final_corners:
                f.write(f"{x}\n")
                f.write(f"{y}\n")
        print(f"Successfully wrote EX2/plots/CENTER.TXT")
    except Exception as e:
        print(f"Failed to write CENTER.TXT: {e}")

def process_map_data(x_robot, y_robot, theta_robot, distance_measured):
    
    # --- 1. SPRING BASED CORRECTION ---
    print("1. Running Spring Optimization...")
    slam = GraphSLAM()
    
    # Build Graph
    for i in range(len(x_robot)):
        slam.add_node(x_robot[i], y_robot[i], theta_robot[i])
        # Only add valid measurements (filter out max range)
        if distance_measured[i] < 200: 
            slam.add_measurement(i, distance_measured[i])
            
    # Relax Graph
    slam.relax_springs()
    
    # Extract Corrected Robot Path
    x_robot_corr = [n['x'] for n in slam.nodes]
    y_robot_corr = [n['y'] for n in slam.nodes]
    
    # --- 2. GENERATE WALL POINTS FROM CORRECTED PATH ---
    # We use the *Optimized* robot position + *Original* sensor reading
    print("2. Generating Wall Points...")
    x_wall = []
    y_wall = []
    
    for i in range(len(x_robot_corr)):
        # Skip invalid measurements for wall generation
        if distance_measured[i] >= 200: continue
            
        sensor_heading = theta_robot[i] + SENSOR_ANGLE_OFFSET
        wx = x_robot_corr[i] + distance_measured[i] * math.cos(sensor_heading)
        wy = y_robot_corr[i] + distance_measured[i] * math.sin(sensor_heading)
        x_wall.append(wx)
        y_wall.append(wy)

    # --- 3. RANSAC EXTRACTION ---
    print("3. Extracting Walls...")
    walls = extract_walls_ransac(x_wall, y_wall)
    
    if not walls:
        print("NO WALLS FOUND! Check your data scale or RANSAC params.")
        return

    # --- 4. GEOMETRIC SORT & PLOTTING (PRESERVED) ---
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
    
    # --- 5. ALIGNMENT ---
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
        
        if m_anchor == float('inf'): current_angle = math.pi / 2
        else: current_angle = math.atan(m_anchor)
            
        rotation_angle = -current_angle
        total_rotation += rotation_angle
        
        sx, sy = rotate_points([start_x], [start_y], rotation_angle)
        start_x, start_y = sx[0], sy[0]
        
        # Apply First Rotation
        all_y_temp = []
        for w in sorted_walls:
            rx, ry = rotate_points(w['x_points'], w['y_points'], rotation_angle)
            w['x_points'] = rx
            w['y_points'] = ry
            all_y_temp.extend(ry)
            
        # Check Upside Down
        map_centroid_y = np.mean(all_y_temp)
        
        # Calculate anchor centroid specifically
        rx_anc, ry_anc = rotate_points(anchor_wall['x_points'], anchor_wall['y_points'], 0) # Already rotated above? No, wait.
        # The points inside sorted_walls are ALREADY updated in the loop above.
        # So we just need the mean of the anchor wall's current Y points.
        anchor_centroid_y = np.mean(anchor_wall['y_points'])
        
        if map_centroid_y < anchor_centroid_y:
            print("Map is upside down. Flipping 180 degrees...")
            rotation_fix = math.pi
            total_rotation += rotation_fix
            
            sx, sy = rotate_points([start_x], [start_y], rotation_fix)
            start_x, start_y = sx[0], sy[0]
            
            for w in sorted_walls:
                rx, ry = rotate_points(w['x_points'], w['y_points'], rotation_fix)
                w['x_points'] = rx
                w['y_points'] = ry
        
        # Re-fit lines
        for w in sorted_walls:
            nm, nc = fit_line_pseudo_inverse(w['x_points'], w['y_points'])
            w['m'] = nm
            w['c'] = nc

    # --- PLOT ---
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

    # Corners
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
                plt.text(mid_x, mid_y, f"{dist:.1f}", fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    calculate_and_save_center(corners_x, corners_y, sorted_walls, start_x, start_y, total_rotation)

    plt.title('Final Map: Spring Corrected', fontsize=16)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
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