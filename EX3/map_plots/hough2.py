import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = r"C:\CS\Robots_mobile\Mobile_Robots\EX3\map_plots\SLAM_PATH.TXT"

X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) 

# --- TUNED PARAMETERS ---
HOUGH_RHO_RES = 1.0
HOUGH_THETA_RES = 3.0    
HOUGH_THRESHOLD = 6
WALL_THICKNESS = 10.0    

SLAM_ITERATIONS = 80
SLAM_LEARNING_RATE = 0.05
SLAM_DIST_THRESH = 40.0  

# --- TRUE ARENA ---
ARENA_PTS = [(-41.5, -30), (-130.5, 59), (-130.5, 207), (59.5, 207), (59.5, -30), (-41.5, -30)]

# --- GLOBAL STATE ---
global_walls = []
global_corners_x = []
global_corners_y = []
global_fig = None
global_ax1 = None # Calculated Map
global_ax2 = None # True Arena

state_rotation = 0    # 0, 90, 180, 270
state_mirrored = False # True/False

random.seed(42)
np.random.seed(42)

# ==============================================================================
# 1. MATH & HELPERS
# ==============================================================================
def rotate_point(x, y, deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return x*c - y*s, x*s + y*c

def transform_points(x_arr, y_arr, rotation_deg, mirror):
    new_x, new_y = [], []
    for x, y in zip(x_arr, y_arr):
        # 1. Mirror (Flip X axis)
        if mirror:
            x = -x
        # 2. Rotate
        rx, ry = rotate_point(x, y, rotation_deg)
        new_x.append(rx)
        new_y.append(ry)
    return new_x, new_y

def fit_line(x, y):
    if len(x) < 2: return 0, 0
    if np.std(y) > np.std(x)*3: return float('inf'), np.mean(x)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def intersect(m1, c1, m2, c2):
    if m1 == m2: return None
    if m1 == float('inf'): return c1, m2*c1 + c2
    if m2 == float('inf'): return c2, m1*c2 + c1
    x = (c2-c1)/(m1-m2)
    return x, m1*x + c1

# ==============================================================================
# 2. SLAM & HOUGH (Standard Logic)
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
        min_dist = SLAM_DIST_THRESH 
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

    def relax(self):
        print(f"  > Relaxing Springs ({SLAM_ITERATIONS} iter)...")
        for _ in range(SLAM_ITERATIONS):
            n_f = {i: [0,0,0] for i in range(len(self.nodes))}
            l_f = {i: [0,0,0] for i in range(len(self.landmarks))}
            for c in self.constraints:
                if c['type'] == 'odo':
                    n1, n2 = self.nodes[c['i']], self.nodes[c['j']]
                    ex, ey = (n1['x']+c['dx'])-n2['x'], (n1['y']+c['dy'])-n2['y']
                    if not n1['fixed']: n_f[c['i']][0]+=ex; n_f[c['i']][1]+=ey; n_f[c['i']][2]+=1
                    if not n2['fixed']: n_f[c['j']][0]-=ex; n_f[c['j']][1]-=ey; n_f[c['j']][2]+=1
                elif c['type'] == 'meas':
                    r, l = self.nodes[c['node_idx']], self.landmarks[c['land_idx']]
                    a = r['theta'] + c['angle_offset']
                    px, py = r['x']+c['dist']*math.cos(a), r['y']+c['dist']*math.sin(a)
                    ex, ey = px-l['x'], py-l['y']
                    if not r['fixed']: n_f[c['node_idx']][0]-=ex; n_f[c['node_idx']][1]-=ey; n_f[c['node_idx']][2]+=1
                    l_f[c['land_idx']][0]+=ex; l_f[c['land_idx']][1]+=ey; l_f[c['land_idx']][2]+=1
            for i, f in n_f.items(): 
                if f[2]: self.nodes[i]['x']+=f[0]/f[2]*SLAM_LEARNING_RATE; self.nodes[i]['y']+=f[1]/f[2]*SLAM_LEARNING_RATE
            for i, f in l_f.items(): 
                if f[2]: self.landmarks[i]['x']+=f[0]/f[2]*SLAM_LEARNING_RATE; self.landmarks[i]['y']+=f[1]/f[2]*SLAM_LEARNING_RATE

def extract_walls(x_all, y_all):
    rem_idx = list(range(len(x_all)))
    walls = []
    thetas = np.deg2rad(np.arange(-90, 90, HOUGH_THETA_RES))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    print(f"  > Hough Transform on {len(x_all)} points...")
    if len(x_all) < HOUGH_THRESHOLD: return []

    while len(rem_idx) > HOUGH_THRESHOLD:
        acc = {}
        cx = np.array([x_all[i] for i in rem_idx])
        cy = np.array([y_all[i] for i in rem_idx])
        for i in range(len(cx)):
            rhos = np.round((cx[i]*cos_t + cy[i]*sin_t)/HOUGH_RHO_RES).astype(int)
            for t_i, r_i in enumerate(rhos):
                key = (r_i, t_i)
                acc[key] = acc.get(key, 0) + 1
        
        if not acc: break
        best = max(acc, key=acc.get)
        if acc[best] < HOUGH_THRESHOLD: break
        
        r_i, t_i = best
        rho = r_i * HOUGH_RHO_RES
        
        # Inliers
        errs = np.abs(cx*cos_t[t_i] + cy*sin_t[t_i] - rho)
        in_local = np.where(errs < WALL_THICKNESS)[0]
        if len(in_local) < HOUGH_THRESHOLD: 
            rem_idx.pop(0); continue

        real_idx = [rem_idx[k] for k in in_local]
        wx = [x_all[i] for i in real_idx]
        wy = [y_all[i] for i in real_idx]
        m, c = fit_line(wx, wy)
        
        walls.append({'m': m, 'c': c, 'x': wx, 'y': wy})
        rem_idx = [i for i in rem_idx if i not in real_idx]
    
    return walls

# ==============================================================================
# 3. INTERACTIVE PLOTTING
# ==============================================================================

def draw_plots():
    global global_ax1, global_ax2, state_rotation, state_mirrored
    
    # --- SUBPLOT 1: CALCULATED MAP ---
    global_ax1.clear()
    global_ax1.set_title(f"Calculated Map\nRotation: {state_rotation}° | Mirrored: {state_mirrored}")
    global_ax1.axis('equal')
    global_ax1.grid(True, linestyle=':', alpha=0.6)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(global_walls)))
    
    # Transform and Plot Walls
    for i, w in enumerate(global_walls):
        # Apply Mirror + Rotate
        tx, ty = transform_points(w['x'], w['y'], state_rotation, state_mirrored)
        
        # Re-fit line for clean drawing
        m, c = fit_line(tx, ty)
        
        # Plot Points
        global_ax1.plot(tx, ty, '.', color=colors[i], markersize=3, alpha=0.6)
        
        # Plot Line
        if len(tx) > 0:
            if m != float('inf'):
                xr = np.linspace(min(tx), max(tx), 10)
                yr = m*xr + c
                global_ax1.plot(xr, yr, '-', color=colors[i], linewidth=2)
            else:
                global_ax1.vlines(c, min(ty), max(ty), colors[i], linewidth=2)

    # Transform and Plot Corners
    if global_corners_x:
        cx, cy = transform_points(global_corners_x, global_corners_y, state_rotation, state_mirrored)
        global_ax1.plot(cx, cy, 'k--', linewidth=1.5, marker='o', markersize=6, markerfacecolor='yellow')
        
        # Add dimensions
        for i in range(len(cx)-1):
            dist = math.sqrt((cx[i+1]-cx[i])**2 + (cy[i+1]-cy[i])**2)
            mx, my = (cx[i]+cx[i+1])/2, (cy[i]+cy[i+1])/2
            global_ax1.text(mx, my, f"{dist:.0f}", fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=0.1))

    # --- SUBPLOT 2: TRUE ARENA (STATIC) ---
    global_ax2.clear()
    global_ax2.set_title("True Arena (Ground Truth)")
    global_ax2.axis('equal')
    global_ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Plot True Arena Polygon
    true_x = [p[0] for p in ARENA_PTS]
    true_y = [p[1] for p in ARENA_PTS]
    
    global_ax2.plot(true_x, true_y, 'k-', linewidth=3, label='Walls')
    global_ax2.fill(true_x, true_y, 'gray', alpha=0.1)
    
    # Label True Dimensions
    for i in range(len(ARENA_PTS)-1):
        p1, p2 = ARENA_PTS[i], ARENA_PTS[i+1]
        dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        global_ax2.text(mx, my, f"{dist:.1f}", color='blue', fontsize=9, fontweight='bold')

    plt.draw()

def on_key(event):
    global state_rotation, state_mirrored
    if event.key == 'r':
        state_rotation = (state_rotation - 90) % 360
        draw_plots()
    elif event.key == 'm':
        state_mirrored = not state_mirrored
        draw_plots()

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
def process_data():
    global global_walls, global_corners_x, global_corners_y, global_fig, global_ax1, global_ax2
    
    try:
        data = np.loadtxt(FILE_NAME, delimiter=',')
    except:
        print("Error loading file."); return

    x_rob = data[:,0]/X_Y_SCALE
    y_rob = data[:,1]/X_Y_SCALE
    th_rob = data[:,2]/THETA_SCALE
    dist = data[:,3]

    # 1. SLAM
    slam = GraphSLAM()
    for i in range(len(x_rob)):
        slam.add_node(x_rob[i], y_rob[i], th_rob[i])
        if dist[i] < 200: slam.add_measurement(i, dist[i])
    slam.relax()

    # 2. Extract Points
    nodes = slam.nodes
    x_w, y_w = [], []
    for i in range(len(nodes)):
        if dist[i] >= 200: continue
        a = nodes[i]['theta'] + SENSOR_ANGLE_OFFSET
        x_w.append(nodes[i]['x'] + dist[i]*math.cos(a))
        y_w.append(nodes[i]['y'] + dist[i]*math.sin(a))

    # 3. Find Walls
    walls = extract_walls(x_w, y_w)
    
    # 4. Sort & Calculate Corners (Base unrotated state)
    all_x = [p for w in walls for p in w['x']]
    all_y = [p for w in walls for p in w['y']]
    cx, cy = np.mean(all_x), np.mean(all_y)
    
    for w in walls:
        wx, wy = np.mean(w['x']), np.mean(w['y'])
        w['angle'] = math.atan2(wy-cy, wx-cx)
    
    walls.sort(key=lambda x: x['angle'])
    global_walls = walls
    
    # Calc corners
    lines = [(w['m'], w['c']) for w in walls]
    corn_x, corn_y = [], []
    if len(lines) >= 3:
        for i in range(len(lines)):
            l1, l2 = lines[i], lines[(i+1)%len(lines)]
            res = intersect(l1[0], l1[1], l2[0], l2[1])
            if res: corn_x.append(res[0]); corn_y.append(res[1])
        if corn_x: corn_x.append(corn_x[0]); corn_y.append(corn_y[0])
    
    global_corners_x = corn_x
    global_corners_y = corn_y
    
    # 5. Initialize Plot
    global_fig, (global_ax1, global_ax2) = plt.subplots(1, 2, figsize=(14, 7))
    global_fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add instructions
    plt.suptitle("SLAM Mapping vs Ground Truth\nControls: 'r' to Rotate (90°), 'm' to Mirror", fontsize=14, fontweight='bold')
    
    draw_plots()
    plt.show()

if __name__ == "__main__":
    process_data()