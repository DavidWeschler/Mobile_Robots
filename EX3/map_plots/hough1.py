import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\map_plots\\SLAM_PATH.TXT"

# Based on your raw data magnitude
X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) 

# --- TUNED PARAMETERS FOR YOUR DATA ---
HOUGH_RHO_RES = 1.0      # 2cm grid for distance
HOUGH_THETA_RES = 3.0    # 3-degree grid (allows for some robot wobble)
HOUGH_THRESHOLD = 6      # NEEDED: Low threshold because you have few points per wall
WALL_THICKNESS = 10.0    # NEEDED: High tolerance to handle the large drift

# Spring Optimization
SLAM_ITERATIONS = 80     # More iterations to fix the heavy drift
SLAM_LEARNING_RATE = 0.05
SLAM_DIST_THRESH = 40.0  # Aggressive loop closure (snap points within 40cm)

# Fix randomness
random.seed(42)
np.random.seed(42)

# ==============================================================================
# 1. SPRING METHOD (GRAPH SLAM) 
# ==============================================================================
class GraphSLAM:
    def __init__(self):
        self.nodes = []       
        self.landmarks = []   
        self.constraints = [] 

    def add_node(self, x, y, theta):
        fixed = (len(self.nodes) == 0)
        self.nodes.append({'x': x, 'y': y, 'theta': theta, 'fixed': fixed})
        if len(self.nodes) > 1:
            prev = self.nodes[-2]
            self.constraints.append({
                'type': 'odo', 'i': len(self.nodes)-2, 'j': len(self.nodes)-1,
                'dx': x - prev['x'], 'dy': y - prev['y']
            })

    def add_measurement(self, node_idx, dist):
        robot = self.nodes[node_idx]
        angle = robot['theta'] + SENSOR_ANGLE_OFFSET
        lx = robot['x'] + dist * math.cos(angle)
        ly = robot['y'] + dist * math.sin(angle)

        best_idx = -1
        min_dist = SLAM_DIST_THRESH # Use tuned threshold
        for i, lm in enumerate(self.landmarks):
            d = math.sqrt((lx - lm['x'])**2 + (ly - lm['y'])**2)
            if d < min_dist:
                min_dist = d
                best_idx = i
        
        if best_idx != -1:
            lm = self.landmarks[best_idx]
            lm['x'] = (lm['x']*lm['n'] + lx)/(lm['n']+1)
            lm['y'] = (lm['y']*lm['n'] + ly)/(lm['n']+1)
            lm['n'] += 1
        else:
            self.landmarks.append({'x': lx, 'y': ly, 'n': 1})
            best_idx = len(self.landmarks) - 1

        self.constraints.append({
            'type': 'meas', 'node_idx': node_idx, 'land_idx': best_idx,
            'dist': dist, 'angle_offset': SENSOR_ANGLE_OFFSET
        })

    def relax_springs(self):
        print(f"  > Relaxing Springs ({SLAM_ITERATIONS} iterations)...")
        for _ in range(SLAM_ITERATIONS):
            n_force = {i: [0.0, 0.0, 0] for i in range(len(self.nodes))}
            l_force = {i: [0.0, 0.0, 0] for i in range(len(self.landmarks))}

            for c in self.constraints:
                if c['type'] == 'odo':
                    n1, n2 = self.nodes[c['i']], self.nodes[c['j']]
                    ex = (n1['x'] + c['dx']) - n2['x']
                    ey = (n1['y'] + c['dy']) - n2['y']
                    if not n1['fixed']:
                        n_force[c['i']][0] += ex; n_force[c['i']][1] += ey; n_force[c['i']][2] += 1
                    if not n2['fixed']:
                        n_force[c['j']][0] -= ex; n_force[c['j']][1] -= ey; n_force[c['j']][2] += 1
                elif c['type'] == 'meas':
                    r = self.nodes[c['node_idx']]
                    l = self.landmarks[c['land_idx']]
                    a = r['theta'] + c['angle_offset']
                    px = r['x'] + c['dist']*math.cos(a)
                    py = r['y'] + c['dist']*math.sin(a)
                    ex = px - l['x']
                    ey = py - l['y']
                    if not r['fixed']:
                        n_force[c['node_idx']][0] -= ex; n_force[c['node_idx']][1] -= ey; n_force[c['node_idx']][2] += 1
                    l_force[c['land_idx']][0] += ex; l_force[c['land_idx']][1] += ey; l_force[c['land_idx']][2] += 1

            for i, f in n_force.items():
                if f[2]>0: self.nodes[i]['x'] += (f[0]/f[2])*SLAM_LEARNING_RATE; self.nodes[i]['y'] += (f[1]/f[2])*SLAM_LEARNING_RATE
            for i, f in l_force.items():
                if f[2]>0: self.landmarks[i]['x'] += (f[0]/f[2])*SLAM_LEARNING_RATE; self.landmarks[i]['y'] += (f[1]/f[2])*SLAM_LEARNING_RATE

# ==============================================================================
# 2. HOUGH TRANSFORM LOGIC
# ==============================================================================

def fit_line_least_squares(x_pts, y_pts):
    if len(x_pts) < 2: return 0, 0
    x = np.array(x_pts)
    y = np.array(y_pts)
    if np.std(y) > np.std(x) * 3: # Vertical
        return float('inf'), np.mean(x)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def extract_walls_hough(x_all, y_all):
    remaining_idx = list(range(len(x_all)))
    found_walls = []
    
    # Pre-calculate Theta values (-90 to 90 degrees)
    thetas = np.deg2rad(np.arange(-90, 90, HOUGH_THETA_RES))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    print(f"  > Running Hough Transform on {len(x_all)} points...")
    if len(x_all) < HOUGH_THRESHOLD:
        print(f"    [!] Not enough points (Need {HOUGH_THRESHOLD}, got {len(x_all)})")
        return []

    while len(remaining_idx) > HOUGH_THRESHOLD:
        
        # 1. Vote
        accumulator = {} 
        curr_x = np.array([x_all[i] for i in remaining_idx])
        curr_y = np.array([y_all[i] for i in remaining_idx])
        
        for i in range(len(curr_x)):
            rhos = curr_x[i] * cos_t + curr_y[i] * sin_t
            rhos_idx = np.round(rhos / HOUGH_RHO_RES).astype(int)
            for t_idx, r_idx in enumerate(rhos_idx):
                key = (r_idx, t_idx)
                accumulator[key] = accumulator.get(key, 0) + 1

        # 2. Peak
        if not accumulator: break
        best_line = max(accumulator, key=accumulator.get)
        votes = accumulator[best_line]
        
        if votes < HOUGH_THRESHOLD: 
            break
            
        r_idx, t_idx = best_line
        best_rho = r_idx * HOUGH_RHO_RES
        
        # 3. Inliers
        dist_errors = np.abs(curr_x * cos_t[t_idx] + curr_y * sin_t[t_idx] - best_rho)
        local_inliers = np.where(dist_errors < WALL_THICKNESS)[0]
        
        if len(local_inliers) < HOUGH_THRESHOLD: 
            remaining_idx.pop(0) 
            continue

        real_indices = [remaining_idx[k] for k in local_inliers]
        
        # 4. Refine & Save
        wall_x = [x_all[i] for i in real_indices]
        wall_y = [y_all[i] for i in real_indices]
        m_final, c_final = fit_line_least_squares(wall_x, wall_y)
        
        found_walls.append({
            'm': m_final, 'c': c_final,
            'x_points': wall_x, 'y_points': wall_y, 'indices': real_indices
        })
        
        print(f"    -> Found Wall with {len(real_indices)} points.")
        remaining_idx = [idx for idx in remaining_idx if idx not in real_indices]

    return found_walls

# ==============================================================================
# 3. HELPERS
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

def calculate_and_save_center(corners_x, corners_y, sorted_walls, start_x, start_y):
    final_corners = []
    if corners_x:
        unique_x = corners_x[:-1] 
        unique_y = corners_y[:-1]
        for cx, cy in zip(unique_x, unique_y):
            dx = cx - start_x
            dy = cy - start_y
            final_corners.append((-dx, -dy))
    
    try:
        with open("CENTER.TXT", "w") as f:
            f.write(f"{len(final_corners)}\n")
            for x, y in final_corners:
                f.write(f"{x}\n")
                f.write(f"{y}\n")
        print(f"Successfully wrote CENTER.TXT")
    except Exception as e:
        print(f"Failed to write CENTER.TXT: {e}")

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================

def process_map_data(x_robot, y_robot, theta_robot, distance_measured):
    
    print("1. Running Spring Optimization...")
    slam = GraphSLAM()
    for i in range(len(x_robot)):
        slam.add_node(x_robot[i], y_robot[i], theta_robot[i])
        if distance_measured[i] < 200: 
            slam.add_measurement(i, distance_measured[i])
    slam.relax_springs()
    
    x_robot_corr = [n['x'] for n in slam.nodes]
    y_robot_corr = [n['y'] for n in slam.nodes]
    
    print("2. Generating Wall Points...")
    x_wall = []
    y_wall = []
    for i in range(len(x_robot_corr)):
        if distance_measured[i] >= 200: continue
        sensor_heading = theta_robot[i] + SENSOR_ANGLE_OFFSET
        wx = x_robot_corr[i] + distance_measured[i] * math.cos(sensor_heading)
        wy = y_robot_corr[i] + distance_measured[i] * math.sin(sensor_heading)
        x_wall.append(wx)
        y_wall.append(wy)

    print(f"   [INFO] Total valid points for detection: {len(x_wall)}")

    print("3. Extracting Walls (Hough Transform)...")
    walls = extract_walls_hough(x_wall, y_wall)
    
    if not walls:
        print("NO WALLS FOUND! Plotting raw points for debug...")
        plt.figure()
        plt.scatter(x_wall, y_wall, s=5)
        plt.plot(x_robot_corr, y_robot_corr, 'r:', alpha=0.3)
        plt.title("Debug: Raw Wall Points (No Lines Detected)")
        plt.show()
        return

    # 4. Sort Walls Geometrically
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
    
    # 5. Alignment
    start_x = x_robot_corr[0]
    start_y = y_robot_corr[0]
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
        if m_anchor == float('inf'): current_angle = math.pi / 2
        else: current_angle = math.atan(m_anchor)
        
        rotation_angle = -current_angle
        
        sx, sy = rotate_points([start_x], [start_y], rotation_angle)
        start_x, start_y = sx[0], sy[0]
        
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
            sx, sy = rotate_points([start_x], [start_y], rotation_fix)
            start_x, start_y = sx[0], sy[0]
            for w in sorted_walls:
                rx, ry = rotate_points(w['x_points'], w['y_points'], rotation_fix)
                w['x_points'] = rx
                w['y_points'] = ry

        for w in sorted_walls:
            nm, nc = fit_line_least_squares(w['x_points'], w['y_points'])
            w['m'] = nm
            w['c'] = nc

    # --- PLOTTING ---
    plt.figure(figsize=(10, 8))
    
    # Plot real arena outline for comparison
    ARENA_PTS = [(-41.5, -30), (-130.5, 59), (-130.5, 207), (59.5, 207), (59.5, -30) ]
    real_x = [p[0] for p in ARENA_PTS]
    real_y = [p[1] for p in ARENA_PTS]
    plt.plot(real_x, real_y, 'k:', linewidth=1, alpha=0.3, label="Ideal Arena")

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
            plt.plot(corners_x, corners_y, 'k--', linewidth=2, marker='o', markersize=8, markerfacecolor='yellow', label='Map')
            for i in range(len(corners_x) - 1):
                p1_x, p1_y = corners_x[i], corners_y[i]
                p2_x, p2_y = corners_x[i+1], corners_y[i+1]
                dist = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)
                mid_x, mid_y = (p1_x + p2_x) / 2, (p1_y + p2_y) / 2
                plt.text(mid_x, mid_y, f"{dist:.1f}", fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    calculate_and_save_center(corners_x, corners_y, sorted_walls, start_x, start_y)
    
    plt.title('Final Map with SLAM', fontsize=16)
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