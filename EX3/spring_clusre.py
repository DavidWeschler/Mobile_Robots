# ==================================================================
#
# Students:
#   - David Weschler       (209736578)
#   - Guy Danin            (205372105)
#   - Benjamin Rosin       (211426598)
#
# ==================================================================

from math import dist
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor

# ==========================================
# 1. Data Parsing
# ==========================================
def parse_robot_log(filename):
    data = np.loadtxt(filename, delimiter=',')
    
    x = data[:, 0] / 10000.0
    y = data[:, 1] / 10000.0
    
    # --- Treat input as Radians directly ---
    # (Assuming the robot logged Radians * 100,000)
    theta_rad = data[:, 2] / 100000.0  
    
    # We no longer calculate theta_deg here, we pass radians directly
    # If you need degrees for plotting labels, you can do np.degrees(theta_rad)
    
    dist = data[:, 3]
    events = data[:, 4]
    
    # Return theta_rad instead of theta_deg
    return x, y, theta_rad, dist, events

# ==========================================
# 2. Loop Closure (Using Perimeter / Path Dist)
# ==========================================
def find_loop_closures(x, y, events, min_perimeter=600.0, max_perimeter=1000.0):
    """
    Links MULTIPLE anchors. Matches current anchors to the same physical 
    anchor from previous laps based on perimeter distance.
    """
    # 1. Calculate cumulative distance driven
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    cumulative_dist = np.cumsum(np.sqrt(dx**2 + dy**2))
    
    anchor_indices =  np.where((events == 1) | (events == 2))[0] #np.where((events == 1))[0] #
    constraints = []
    
    print(f"Finding all anchor links across laps (Lap size: {min_perimeter}-{max_perimeter}cm)...")
    
    for i in range(len(anchor_indices)):
        idx_current = anchor_indices[i]
        
        # Look backwards through history to find the matching anchor from 1 lap ago
        for j in range(i - 1, -1, -1):
            idx_past = anchor_indices[j]
            
            dist_driven = cumulative_dist[idx_current] - cumulative_dist[idx_past]
            
            # If the distance between these two hits is roughly ONE LAP, 
            # they are the exact same physical anchor on the track!
            if min_perimeter <= dist_driven <= max_perimeter:
                print(f"  -> Elastic Link: Step {idx_current} <=> Step {idx_past} (Lap Dist: {dist_driven:.1f}cm)")
                constraints.append((idx_current, idx_past))
                
                # We found the match from the previous lap, move to the next hit
                break 
            
            # If we've looked back further than the max perimeter, stop looking 
            # (We only need to link to the immediate previous lap to build the chain)
            if dist_driven > max_perimeter:
                break
                
    return constraints

# ==========================================
# 3. Graph SLAM Optimization
# ==========================================
def optimize_pose_graph(x, y, theta, events):
    
    # A. Find the Loops using Perimeter Logic
    # We allow perimeter between 6m (600cm) and 10m (1000cm)
    loop_constraints = find_loop_closures(x, y, events)
    
    # B. Pre-calculate relative odometry
    dx_odom = np.diff(x)
    dy_odom = np.diff(y)
    
    # C. The Solver Function
    def graph_energy(params):
        n = len(params) // 3
        curr_x = params[:n]
        curr_y = params[n:2*n]
        # curr_theta = params[2*n:] 
        
        residuals = []
        
        # 1. Odometry Springs (Maintain Shape)
        est_dx = np.diff(curr_x)
        est_dy = np.diff(curr_y)
        
        w_odom = 1.0
        residuals.extend((est_dx - dx_odom) * w_odom)
        residuals.extend((est_dy - dy_odom) * w_odom)
        
        # 2. Loop Closure Springs
        w_loop = 50.0 # Strong spring to snap the loop
        for (idx_now, idx_old) in loop_constraints:
            diff_x = curr_x[idx_now] - curr_x[idx_old]
            diff_y = curr_y[idx_now] - curr_y[idx_old]
            
            residuals.append(diff_x * w_loop)
            residuals.append(diff_y * w_loop)
            
        # 3. Anchor Pin (Pin start to 0,0)
        residuals.append(curr_x[0] * 100.0)
        residuals.append(curr_y[0] * 100.0)
            
        return np.array(residuals)

    # D. Run Optimization
    print("Optimizing Pose Graph...")
    initial_params = np.concatenate([x, y, theta])
    result = least_squares(graph_energy, initial_params, method='trf')
    
    n = len(x)
    return result.x[:n], result.x[n:2*n], result.x[2*n:]

# ==========================================
# 4. Wall Extraction
# ==========================================
def get_walls(x, y, theta_rad, dist):
    """
    Extracts walls using RANSAC + SVD, then extends them to form a closed loop.
    """
    sensor_angle = theta_rad + (np.pi / 2) # Sensor is on the Left (+90 deg)
    
    wx = x + dist * np.cos(sensor_angle)
    wy = y + dist * np.sin(sensor_angle)        
    
    walls = []
    
    # Create a unified (N, 2) array of points
    points = np.column_stack((wx, wy))
    remaining_points = points.copy()
    

    min_inliers = 5     
    distance_threshold = 2.0 
    num_iterations = 1000   
    
    for _ in range(10):
        if len(remaining_points) < min_inliers:
            break
            
        best_inlier_mask = None
        max_votes = 0
        
        for _ in range(num_iterations):
            idx = np.random.choice(len(remaining_points), 2, replace=False)
            p1 = remaining_points[idx[0]]
            p2 = remaining_points[idx[1]]
            
            vec = p2 - p1
            norm = np.linalg.norm(vec)
            if norm < 0.1: continue 
            
            normal = np.array([-vec[1], vec[0]]) / norm
            diffs = remaining_points - p1
            distances = np.abs(np.dot(diffs, normal))
            
            current_inlier_mask = distances < distance_threshold
            vote_count = np.sum(current_inlier_mask)
            
            if vote_count > max_votes:
                max_votes = vote_count
                best_inlier_mask = current_inlier_mask
                
        if max_votes < min_inliers:
            break 
            
        wall_points = remaining_points[best_inlier_mask]
        
        # SVD FIT
        mean = np.mean(wall_points, axis=0)
        centered = wall_points - mean
        U, S, Vt = np.linalg.svd(centered)
        direction = Vt[0]
        
        projections = np.dot(centered, direction)
        sort_idx = np.argsort(projections)
        sorted_wall_points = wall_points[sort_idx]
        
        walls.append(sorted_wall_points)
        remaining_points = remaining_points[~best_inlier_mask]

    # --- PHASE 4: CONNECT & CLEAN (NEW LOGIC) ---
    if len(walls) < 3: 
        return wx, wy, walls # Need at least 3 walls to make a loop

    # 1. Convert point clouds to abstract "Line Objects"
    line_objs = []
    for w in walls:
        # Recalculate centroid and direction for the intersection math
        start_pt = w[0]
        end_pt = w[-1]
        centroid = np.mean(w, axis=0)
        direction = (end_pt - start_pt) / np.linalg.norm(end_pt - start_pt)
        
        line_objs.append({
            'centroid': centroid,
            'dir': direction,
            'orig_start': start_pt, # Keep track of original size
            'orig_end': end_pt
        })

    # 2. Sort Angularly (Find the loop order)
    # Calculate approximate room center
    room_center = np.mean([obj['centroid'] for obj in line_objs], axis=0)
    
    for obj in line_objs:
        dx = obj['centroid'][0] - room_center[0]
        dy = obj['centroid'][1] - room_center[1]
        obj['angle'] = np.arctan2(dy, dx)
        
    sorted_lines = sorted(line_objs, key=lambda x: x['angle'])
    
    # 3. Intersect Neighbors
    final_connected_walls = []
    num = len(sorted_lines)
    
    def get_intersection(p1, v1, p2, v2):
        # Solves intersection of two lines: p1 + t*v1 = p2 + u*v2
        det = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(det) < 1e-6: return None # Parallel lines
        diff = p2 - p1
        t = (diff[0]*v2[1] - diff[1]*v2[0]) / det
        return p1 + t * v1

    # We iterate through the sorted walls and try to connect i to (i+1)
    for i in range(num):
        curr = sorted_lines[i]
        next_line = sorted_lines[(i + 1) % num]
        
        # Find the corner where these two walls meet
        corner = get_intersection(curr['centroid'], curr['dir'], 
                                  next_line['centroid'], next_line['dir'])
        
        # Validation: Is the corner reasonable?
        # If the corner is 5 meters away from the wall's data, it's likely noise/parallel
        if corner is not None:
            dist_to_curr = np.linalg.norm(corner - curr['centroid'])
            dist_to_next = np.linalg.norm(corner - next_line['centroid'])
            
            # Threshold: Allow extending up to 200cm (2 meters) to find a corner
            if dist_to_curr < 200 and dist_to_next < 200:
                curr['end_corner'] = corner
                next_line['start_corner'] = corner
            else:
                # Corner is too far, these walls probably don't touch
                pass

    # 4. Reconstruct Valid Segments
    # Only keep walls that successfully connected on BOTH sides (or at least one side)
    for obj in sorted_lines:
        # If we found corners, use them. Otherwise, fall back to original points.
        # Strict Mode: Only add if we have BOTH corners (Start and End)
        if 'start_corner' in obj and 'end_corner' in obj:
            # Create a 2-point array [Start, End] representing the clean wall
            clean_wall = np.array([obj['start_corner'], obj['end_corner']])
            final_connected_walls.append(clean_wall)
        
        # (Optional) Loose Mode: If you want to keep walls that connected on just one side:
        # elif 'start_corner' in obj:
        #     clean_wall = np.array([obj['start_corner'], obj['orig_end']])
        #     final_connected_walls.append(clean_wall)
        
    # If the filtering removed everything (e.g. erratic data), fall back to raw walls
    if len(final_connected_walls) == 0:
        return wx, wy, walls
            
    return wx, wy, final_connected_walls

# ==========================================
# 5. Save Results (ADDED)
# ==========================================
def save_results(path_x, path_y, walls):
    print("Saving output files...")
    
    # 1. Save Robot Path
    # Format: X, Y
    np.savetxt('EX3/txt_files/robot_path.txt', np.column_stack((path_x, path_y)), 
               fmt='%.4f', header='X_cm,Y_cm', comments='', delimiter=',')
    print(" -> robot_path.txt saved.")

    # 2. Save Clean Walls
    # Format: StartX, StartY, EndX, EndY, Length
    wall_data = []
    for w in walls:
        start = w[0]
        end = w[-1]
        length = np.linalg.norm(end - start)
        wall_data.append([start[0], start[1], end[0], end[1], length])
        
    if len(wall_data) > 0:
        np.savetxt('EX3/txt_files/clean_walls.txt', np.array(wall_data), 
                   fmt='%.4f', header='X_Start,Y_Start,X_End,Y_End,Length_cm', comments='', delimiter=',')
        print(" -> clean_walls.txt saved.")
    else:
        print(" -> No walls to save.")

# ==========================================
# 6. Execution & Plotting
# ==========================================
f2 = r'EX3\TRACK_LOG_de_best.TXT'
# f4 = r'C:\Users\isrgd\robots\Mobile_Robots\EX3\TRACK_LOG_18_M.TXT'

# 1. Load Data
x_raw, y_raw, th_raw, dist, events = parse_robot_log(f2)

# 2. Run Graph SLAM (Optimizes the WHOLE path)
x_opt, y_opt, th_opt = optimize_pose_graph(x_raw, y_raw, th_raw, events)

# 3. Find Loops to define the "Single Lap" range
loops = find_loop_closures(x_raw, y_raw, events)

# --- NEW LOGIC: EXTRACT SINGLE LOOP ---
if len(loops) > 0:
    # Sort loops by their START point (idx_past) to find the very first lap
    first_loop = min(loops, key=lambda item: item[1]) 
    
    # Define the range
    start_idx = first_loop[1]
    end_idx = first_loop[0]
    
    print(f"Mapping using ONLY the first loop: Index {start_idx} to {end_idx}")
    
    # Slice the OPTIMIZED data (with small buffer)
    x_map = x_opt[start_idx : end_idx] # +10
    y_map = y_opt[start_idx : end_idx]
    th_map = th_opt[start_idx : end_idx]
    dist_map = dist[start_idx : end_idx]
    
else:
    print("No loop found! Mapping using the entire path.")
    x_map, y_map, th_map, dist_map = x_opt, y_opt, th_opt, dist

# 4. Get Walls (Using ONLY the sliced Single Loop data)
wx, wy, walls = get_walls(x_map, y_map, th_map, dist_map)

# 5. Save Results (ADDED)
save_results(x_map, y_map, walls)

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(14, 6))

# --- Subplot 1: Trajectory ---
plt.subplot(1, 2, 1)
plt.title("Pose Graph Optimization")
plt.plot(x_raw, y_raw, 'r--', label='Raw Odom', alpha=0.3)
plt.plot(x_opt, y_opt, 'b-', label='Optimized Path', lw=1, alpha=0.5)

# Highlight the specific loop we used for mapping
plt.plot(x_map, y_map, 'g-', lw=2, label='Mapping Segment')

# Plot Anchors & Links
raw_anchor_indices = np.where((events == 1) | (events == 2))[0]
plt.scatter(x_raw[raw_anchor_indices], y_raw[raw_anchor_indices], 
            c='orange', s=30, zorder=4, label='Anchors')

for (idx_now, idx_old) in loops:
    plt.plot([x_opt[idx_now], x_opt[idx_old]], [y_opt[idx_now], y_opt[idx_old]], 'g-', lw=1, alpha=0.3)

plt.legend()
plt.axis('equal')

# --- Subplot 2: Map with Dimensions ---
plt.subplot(1, 2, 2)
plt.title("Final Map with Dimensions (cm)")

# 1. Plot Sensor Cloud (Gray)
theta_rad = np.radians(th_map)
sensor_angle = theta_rad + (np.pi / 2)
map_wx = x_map + dist_map * np.cos(sensor_angle)
map_wy = y_map + dist_map * np.sin(sensor_angle)
plt.scatter(map_wx, map_wy, s=1, c='gray', alpha=0.3, label='Sensor Cloud')

# 2. Plot Walls & Calculate Lengths
for i, w in enumerate(walls):
    # Plot the red wall line
    plt.plot(w[:,0], w[:,1], 'r-', lw=3)
    
    # Calculate Length (Euclidean Distance between start and end)
    start_pt = w[0]
    end_pt = w[-1]
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    length = np.sqrt(dx**2 + dy**2)
    
    # Calculate Midpoint for text label
    mid_x = (start_pt[0] + end_pt[0]) / 2
    mid_y = (start_pt[1] + end_pt[1]) / 2
    
    # Add text label with white background box for readability
    plt.text(mid_x, mid_y, f"{length:.1f}cm", 
             color='blue', fontsize=10, fontweight='bold', ha='center',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()