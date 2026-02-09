import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Paths
FILE_PATH = r'C:\CS\Robots_mobile\Mobile_Robots\EX3\GUY_TRACK_LOG.TXT'
CLEANED_FILE = 'SLAM_CLEANED.TXT'

# Scaling factors (From Code 2)
X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) 

# SLAM Parameters
SLAM_ITERATIONS = 150 
SLAM_LEARNING_RATE = 0.05 
SLAM_DIST_THRESH = 1.0 

# RANSAC Parameters
RANSAC_MAX_DIST = 8.0 
RANSAC_MIN_POINTS = 10 
RANSAC_ITERATIONS = 2000 

# Reproducibility
random.seed(42)
np.random.seed(42)

# ==============================================================================
# PART 1: DATA LOADING & ANCHOR CORRECTION
# ==============================================================================

def correct_drift_using_anchors(data):
    """
    Uses anchor points (Marker=1) to correct odometry drift.
    Assumption: All anchor points represent the EXACT same physical (x,y,theta).
    Method: Linear error distribution (Piecewise Linear Interpolation).
    """
    # 1. Extract Columns
    # Format: [Col0, Col1, Col2, Col3, Marker]
    rx = data[:, 0]
    ry = data[:, 1]
    rtheta = data[:, 2]
    markers = data[:, 4]

    # 2. Find Anchors
    anchor_indices = np.where(markers == 1)[0]
    
    if len(anchor_indices) < 2:
        print(" ! Not enough anchors for drift correction. Returning trimmed data only.")
        return data[anchor_indices[0]:anchor_indices[-1]+1]

    print(f" > Found {len(anchor_indices)} anchor points. Correcting drift...")

    # 3. Initialize Corrected Arrays with Original Data
    # We will slice strictly from First Anchor to Last Anchor
    start_global = anchor_indices[0]
    end_global = anchor_indices[-1]
    
    # Work on a copy of the slice
    segment_data = data[start_global : end_global+1].copy()
    
    # Re-find anchor indices relative to this new slice
    # (The first row of segment_data is now index 0, which is the first anchor)
    rel_anchors = anchor_indices - start_global

    # 4. Perform Correction Segment by Segment
    # We assume the First Anchor (Index 0) is the "Truth".
    # We force subsequent anchors to match the First Anchor's X, Y, Theta.
    
    truth_x = segment_data[0, 0]
    truth_y = segment_data[0, 1]
    truth_th = segment_data[0, 2]

    # Iterate through pairs of anchors (e.g., Anchor 1 -> Anchor 2, Anchor 2 -> Anchor 3)
    for i in range(len(rel_anchors) - 1):
        idx_start = rel_anchors[i]
        idx_end = rel_anchors[i+1]
        
        # The position of the next anchor currently
        curr_end_x = segment_data[idx_end, 0]
        curr_end_y = segment_data[idx_end, 1]
        curr_end_th = segment_data[idx_end, 2]

        # Calculate the Error (Drift) at the end of this segment
        error_x = curr_end_x - truth_x
        error_y = curr_end_y - truth_y
        error_th = curr_end_th - truth_th

        # Distribute this error linearly across the segment
        segment_len = idx_end - idx_start
        
        for k in range(1, segment_len + 1):
            # Calculate weight (0 at start, 1 at end)
            w = k / segment_len
            
            # Apply correction to the row in the segment
            # We subtract the accumulated drift proportional to distance traveled (index)
            # Note: We apply this cumulatively to the rest of the array to keep continuity
            # BUT efficient way is to correct just this segment, then shift the rest.
            # Simpler approach here: Correct the points in this interval.
            
            # However, since drift accumulates, if we fix index_end to match truth,
            # the "start" of the NEXT segment is now at "truth".
            # So we only need to warp the points *between* idx_start and idx_end.
            
            row_idx = idx_start + k
            
            # Simple linear subtraction of error
            segment_data[row_idx, 0] -= (error_x * w)
            segment_data[row_idx, 1] -= (error_y * w)
            segment_data[row_idx, 2] -= (error_th * w)

        # IMPORTANT: Since we modified the array in place up to idx_end, 
        # the point at idx_end is now exactly at (truth_x, truth_y).
        # However, the points AFTER idx_end (the next segment) were recorded 
        # relative to the *uncorrected* position. They need to be shifted 
        # rigidly by the total error found at idx_end.
        
        if i < len(rel_anchors) - 2:
            next_start = idx_end + 1
            segment_data[next_start:, 0] -= error_x
            segment_data[next_start:, 1] -= error_y
            segment_data[next_start:, 2] -= error_th

    return segment_data

def process_and_save_data():
    try:
        raw_data = np.genfromtxt(FILE_PATH, delimiter=',')
        print(f"Loaded {len(raw_data)} rows.")
    except Exception as e:
        print(f"Error loading {FILE_PATH}: {e}")
        sys.exit()

    # Apply Correction
    corrected_data = correct_drift_using_anchors(raw_data)
    
    # Save
    np.savetxt(CLEANED_FILE, corrected_data, delimiter=',', fmt='%.2f')
    print(f"Saved Corrected Data ({len(corrected_data)} rows) to {CLEANED_FILE}")

    # --- PLOT COMPARISON ---
    # Show Raw vs Corrected Path
    plt.figure(figsize=(10, 5))
    
    # Plot Original (Trimmed to same range for fair comparison)
    # We find indices again just for plotting original slice
    markers = raw_data[:, 4]
    anchors = np.where(markers == 1)[0]
    if len(anchors) >= 2:
        orig_slice = raw_data[anchors[0]:anchors[-1]+1]
        plt.plot(orig_slice[:, 0], orig_slice[:, 1], 'r--', alpha=0.5, label='Raw (Drifting)')
    
    # Plot Corrected
    plt.plot(corrected_data[:, 0], corrected_data[:, 1], 'b-', linewidth=1, label='Anchor Corrected')
    
    # Plot Anchor Points
    start_x, start_y = corrected_data[0, 0], corrected_data[0, 1]
    plt.scatter([start_x], [start_y], c='k', marker='x', s=100, label='Anchor (Fixed)')
    
    plt.title("Step 1: Loop Closure using Anchors")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    print("Displaying Correction Plot. Close window to continue to SLAM...")
    plt.show()

# ==============================================================================
# PART 2: SLAM ALGORITHMS (Spring Mesh)
# ==============================================================================

class GraphSLAM:
    def __init__(self):
        self.nodes = []       # Robot poses
        self.landmarks = []   # Observed features (walls/corners)
        self.constraints = [] # Springs (Odometry or Measurement)

    def add_node(self, x, y, theta):
        # First node is fixed to anchor the map
        fixed = (len(self.nodes) == 0)
        self.nodes.append({'x': x, 'y': y, 'theta': theta, 'fixed': fixed})
        
        # Add odometry constraint (spring between current and previous node)
        if len(self.nodes) > 1:
            prev = self.nodes[-2]
            self.constraints.append({
                'type': 'odo', 
                'i': len(self.nodes)-2, 
                'j': len(self.nodes)-1,
                'dx': x - prev['x'], 
                'dy': y - prev['y']
            })

    def add_measurement(self, node_idx, dist):
        # Project measurement into world coordinates
        robot = self.nodes[node_idx]
        angle = robot['theta'] + SENSOR_ANGLE_OFFSET
        lx = robot['x'] + dist * math.cos(angle)
        ly = robot['y'] + dist * math.sin(angle)

        # Data Association: Find best existing landmark
        best_idx = -1
        min_dist = SLAM_DIST_THRESH
        
        for i, lm in enumerate(self.landmarks):
            d = math.sqrt((lx - lm['x'])**2 + (ly - lm['y'])**2)
            if d < min_dist:
                min_dist = d
                best_idx = i
        
        # Update existing landmark or create new one
        if best_idx != -1:
            lm = self.landmarks[best_idx]
            # Simple weighted average update
            lm['x'] = (lm['x']*lm['n'] + lx)/(lm['n']+1)
            lm['y'] = (lm['y']*lm['n'] + ly)/(lm['n']+1)
            lm['n'] += 1
        else:
            self.landmarks.append({'x': lx, 'y': ly, 'n': 1})
            best_idx = len(self.landmarks) - 1

        # Add measurement constraint (spring between node and landmark)
        self.constraints.append({
            'type': 'meas', 
            'node_idx': node_idx, 
            'land_idx': best_idx,
            'dist': dist, 
            'angle_offset': SENSOR_ANGLE_OFFSET
        })

    def relax_springs(self):
        print(f" > Relaxing SLAM Springs ({SLAM_ITERATIONS} iterations)...")
        for _ in range(SLAM_ITERATIONS):
            # Accumulators for forces
            n_force = {i: [0.0, 0.0, 0] for i in range(len(self.nodes))}
            l_force = {i: [0.0, 0.0, 0] for i in range(len(self.landmarks))}

            for c in self.constraints:
                if c['type'] == 'odo':
                    n1, n2 = self.nodes[c['i']], self.nodes[c['j']]
                    target_x = n1['x'] + c['dx']
                    target_y = n1['y'] + c['dy']
                    ex, ey = target_x - n2['x'], target_y - n2['y']
                    
                    if not n1['fixed']:
                        n_force[c['i']][0] += -ex; n_force[c['i']][1] += -ey; n_force[c['i']][2] += 1
                    if not n2['fixed']:
                        n_force[c['j']][0] += ex; n_force[c['j']][1] += ey; n_force[c['j']][2] += 1

                elif c['type'] == 'meas':
                    r = self.nodes[c['node_idx']]
                    l = self.landmarks[c['land_idx']]
                    a = r['theta'] + c['angle_offset']
                    
                    proj_x = r['x'] + c['dist']*math.cos(a)
                    proj_y = r['y'] + c['dist']*math.sin(a)
                    ex, ey = proj_x - l['x'], proj_y - l['y']

                    if not r['fixed']:
                        n_force[c['node_idx']][0] -= ex; n_force[c['node_idx']][1] -= ey; n_force[c['node_idx']][2] += 1
                    l_force[c['land_idx']][0] += ex; l_force[c['land_idx']][1] += ey; l_force[c['land_idx']][2] += 1

            # Apply forces
            for i, f in n_force.items():
                if f[2]>0: 
                    self.nodes[i]['x'] += (f[0]/f[2])*SLAM_LEARNING_RATE
                    self.nodes[i]['y'] += (f[1]/f[2])*SLAM_LEARNING_RATE
            
            for i, f in l_force.items():
                if f[2]>0:
                    self.landmarks[i]['x'] += (f[0]/f[2])*SLAM_LEARNING_RATE
                    self.landmarks[i]['y'] += (f[1]/f[2])*SLAM_LEARNING_RATE

# ==============================================================================
# PART 3: RANSAC & MATH
# ==============================================================================

def fit_line_least_squares(x_pts, y_pts):
    x, y = np.array(x_pts), np.array(y_pts)
    if len(x) < 2: return 0, 0
    if np.std(x) < 0.01 * np.std(y): return float('inf'), np.mean(x)
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def extract_walls_ransac(x_all, y_all):
    remaining_idx = list(range(len(x_all)))
    found_walls = []
    
    print(f" > RANSAC: Processing {len(x_all)} points...")

    while len(remaining_idx) > RANSAC_MIN_POINTS:
        best_inliers = []
        
        for _ in range(RANSAC_ITERATIONS):
            if len(remaining_idx) < 2: break
            
            idx1, idx2 = random.sample(remaining_idx, 2)
            x1, y1 = x_all[idx1], y_all[idx1]
            x2, y2 = x_all[idx2], y_all[idx2]
            
            A = y1 - y2
            B = x2 - x1
            C = -A * x1 - B * y1
            denom = math.sqrt(A**2 + B**2)
            if denom == 0: continue

            curr_inliers = []
            for idx in remaining_idx:
                dist = abs(A * x_all[idx] + B * y_all[idx] + C) / denom
                if dist < RANSAC_MAX_DIST:
                    curr_inliers.append(idx)
            
            if len(curr_inliers) > len(best_inliers):
                best_inliers = curr_inliers
        
        if len(best_inliers) < RANSAC_MIN_POINTS: 
            break
        
        wall_x = [x_all[i] for i in best_inliers]
        wall_y = [y_all[i] for i in best_inliers]
        m, c = fit_line_least_squares(wall_x, wall_y)
        
        found_walls.append({
            'm': m, 'c': c, 
            'x_points': wall_x, 'y_points': wall_y, 
            'indices': best_inliers
        })
        
        remaining_idx = [i for i in remaining_idx if i not in best_inliers]
        
    return found_walls

def find_intersection(m1, c1, m2, c2):
    if m1 == float('inf') and m2 == float('inf'): return None
    if m1 == float('inf'): return c1, m2 * c1 + c2
    if m2 == float('inf'): return c2, m1 * c2 + c1
    if abs(m1 - m2) < 1e-5: return None 
    
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

def rotate_points(x_arr, y_arr, angle_rad):
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    nx = [x * cos_a - y * sin_a for x, y in zip(x_arr, y_arr)]
    ny = [x * sin_a + y * cos_a for x, y in zip(x_arr, y_arr)]
    return nx, ny

# ==============================================================================
# PART 4: VISUALIZATION
# ==============================================================================

class InteractiveMapVisualizer:
    def __init__(self, sorted_walls, corners_x, corners_y):
        self.walls = sorted_walls
        self.corners_x = corners_x
        self.corners_y = corners_y
        self.mirror_factor = 1 
        self.rotation_angle = 0 
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.plot_ideal()
        self.update_map_plot()
        print("\n[Controls] Press 'r' to Rotate 90 deg, 'm' to Mirror.")
        plt.show()

    def on_key(self, event):
        if event.key == 'r':
            self.rotation_angle += math.pi / 2
            self.update_map_plot()
        elif event.key == 'm':
            self.mirror_factor *= -1
            self.update_map_plot()

    def transform_points(self, x_list, y_list):
        x_m = [x * self.mirror_factor for x in x_list]
        y_m = y_list
        return rotate_points(x_m, y_m, self.rotation_angle)

    def plot_ideal(self):
        self.ax1.set_title("Ideal Arena Reference", fontsize=14)
        ARENA_PTS = [(-41.5, -30), (-130.5, 59), (-130.5, 207), (59.5, 207), (59.5, -30)]
        rx = [p[0] for p in ARENA_PTS]
        ry = [p[1] for p in ARENA_PTS]
        self.ax1.plot(rx + [rx[0]], ry + [ry[0]], 'k-', linewidth=2)
        self.ax1.fill(rx + [rx[0]], ry + [ry[0]], alpha=0.1, color='gray')
        self.ax1.axis('equal')
        self.ax1.grid(True)

    def update_map_plot(self):
        self.ax2.clear()
        title_str = f"Generated Map (Rot: {math.degrees(self.rotation_angle):.0f}°, Mir: {self.mirror_factor})"
        self.ax2.set_title(title_str, fontsize=14)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.walls)))

        for i, w in enumerate(self.walls):
            tx, ty = self.transform_points(w['x_points'], w['y_points'])
            self.ax2.plot(tx, ty, '.', color=colors[i], markersize=3, alpha=0.6)
            
        if self.corners_x:
            cx_t, cy_t = self.transform_points(self.corners_x, self.corners_y)
            cx_plot = cx_t + [cx_t[0]]
            cy_plot = cy_t + [cy_t[0]]
            
            # Draw lines between corners
            self.ax2.plot(cx_plot, cy_plot, 'k--', alpha=0.5)
            
            for i in range(len(cx_t)):
                p1 = (cx_plot[i], cy_plot[i])
                p2 = (cx_plot[i+1], cy_plot[i+1])
                dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                self.ax2.text(mid[0], mid[1], f"{dist:.1f}", fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

        self.ax2.axis('equal')
        self.ax2.grid(True)
        self.fig.canvas.draw()

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_full_pipeline():
    # --- STEP 1: LOAD & CORRECT DRIFT ---
    process_and_save_data()
    
    # --- STEP 2: LOAD CORRECTED DATA & RUN SLAM ---
    print("\nStarting SLAM on corrected data...")
    data = np.loadtxt(CLEANED_FILE, delimiter=',')
    
    x_raw = data[:, 0] / X_Y_SCALE
    y_raw = data[:, 1] / X_Y_SCALE
    theta_raw = data[:, 2] / THETA_SCALE
    dist_meas = data[:, 3]

    # Spring Graph SLAM
    slam = GraphSLAM()
    for i in range(len(x_raw)):
        slam.add_node(x_raw[i], y_raw[i], theta_raw[i])
        if dist_meas[i] < 200: 
            slam.add_measurement(i, dist_meas[i])
    
    slam.relax_springs()

    # Calculate Wall Points
    x_robot_corr = [n['x'] for n in slam.nodes]
    y_robot_corr = [n['y'] for n in slam.nodes]
    x_wall, y_wall = [], []

    for i in range(len(x_robot_corr)):
        if dist_meas[i] >= 200: continue
        heading = theta_raw[i] + SENSOR_ANGLE_OFFSET
        wx = x_robot_corr[i] + dist_meas[i] * math.cos(heading)
        wy = y_robot_corr[i] + dist_meas[i] * math.sin(heading)
        x_wall.append(wx)
        y_wall.append(wy)

    # RANSAC
    walls = extract_walls_ransac(x_wall, y_wall)
    
    if not walls:
        print("No walls found!")
        return

    # Sort & Find Corners
    cx = np.mean([p for w in walls for p in w['x_points']])
    cy = np.mean([p for w in walls for p in w['y_points']])
    
    for w in walls:
        mid_x = np.mean(w['x_points'])
        mid_y = np.mean(w['y_points'])
        w['angle'] = math.atan2(mid_y - cy, mid_x - cx)
        
    sorted_walls = sorted(walls, key=lambda x: x['angle'])
    
    lines = [(w['m'], w['c']) for w in sorted_walls]
    corn_x, corn_y = [], []
    
    if len(lines) >= 3:
        for i in range(len(lines)):
            l1 = lines[i]
            l2 = lines[(i+1) % len(lines)]
            res = find_intersection(l1[0], l1[1], l2[0], l2[1])
            if res:
                corn_x.append(res[0])
                corn_y.append(res[1])

    # Visualize
    InteractiveMapVisualizer(sorted_walls, corn_x, corn_y)

if __name__ == "__main__":
    run_full_pipeline()