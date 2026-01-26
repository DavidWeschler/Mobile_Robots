import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --- Configuration ---
FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\map_plots\\SLAM_PATH.TXT"
X_Y_SCALE = 10000.0
THETA_SCALE = 100000.0
SENSOR_ANGLE_OFFSET = (math.pi / 2.0)

# Optimization Constraints
ITERATIONS = 50       # How many times to relax the springs
LEARNING_RATE = 0.1   # How much nodes move per step (Stiffness)
DIST_THRESHOLD = 15.0 # Max distance to associate a point with an existing landmark

# Fix randomness
random.seed(42)
np.random.seed(42)

# --- 1. Math Helpers ---

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def rotate_point(x, y, angle):
    """ Rotates a point by a specific angle (radians) """
    nx = x * math.cos(angle) - y * math.sin(angle)
    ny = x * math.sin(angle) + y * math.cos(angle)
    return nx, ny

# --- 2. Spring-Based Optimization (Graph SLAM) ---

class GraphSLAM:
    def __init__(self):
        self.nodes = []       # Robot Poses: [{'x', 'y', 'theta', 'fixed': False}, ...]
        self.landmarks = []   # Wall Points: [{'x', 'y', 'count'}]
        self.constraints = [] # Springs: [{'type': 'odo'/'meas', 'node_idx', 'landmark_idx', 'dx', 'dy'}]

    def add_robot_node(self, x, y, theta):
        # The first node is fixed (anchor) so the whole map doesn't float away
        is_fixed = (len(self.nodes) == 0)
        self.nodes.append({'x': x, 'y': y, 'theta': theta, 'fixed': is_fixed})

        # Add Odometry Constraint (Spring to previous node)
        if len(self.nodes) > 1:
            prev_idx = len(self.nodes) - 2
            curr_idx = len(self.nodes) - 1
            prev = self.nodes[prev_idx]
            
            # The "rest length" of the spring is the odometry delta
            dx = x - prev['x']
            dy = y - prev['y']
            dtheta = theta - prev['theta']
            
            self.constraints.append({
                'type': 'odometry',
                'i': prev_idx,
                'j': curr_idx,
                'dx': dx,
                'dy': dy,
                'dtheta': dtheta
            })

    def add_measurement(self, robot_idx, range_dist):
        """ Adds a spring between the robot and a landmark (wall point) """
        robot = self.nodes[robot_idx]
        
        # Calculate where the wall *should* be based on current robot pose
        sensor_heading = robot['theta'] + SENSOR_ANGLE_OFFSET
        lx = robot['x'] + range_dist * math.cos(sensor_heading)
        ly = robot['y'] + range_dist * math.sin(sensor_heading)

        # DATA ASSOCIATION: Check if this matches an existing landmark
        best_idx = -1
        min_dist = DIST_THRESHOLD

        for i, lm in enumerate(self.landmarks):
            dist = get_distance((lx, ly), (lm['x'], lm['y']))
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # If match found, connect to existing landmark (Loop Closure!)
        if best_idx != -1:
            # Update landmark position average
            lm = self.landmarks[best_idx]
            lm['x'] = (lm['x'] * lm['count'] + lx) / (lm['count'] + 1)
            lm['y'] = (lm['y'] * lm['count'] + ly) / (lm['count'] + 1)
            lm['count'] += 1
        else:
            # Create new landmark
            self.landmarks.append({'x': lx, 'y': ly, 'count': 1})
            best_idx = len(self.landmarks) - 1

        # Add Measurement Constraint (Spring)
        self.constraints.append({
            'type': 'measurement',
            'node_idx': robot_idx,
            'land_idx': best_idx,
            'dist': range_dist,
            'angle_offset': SENSOR_ANGLE_OFFSET
        })

    def relax_springs(self):
        """ The Core 'Spring' Physics Engine """
        print(f"Relaxing springs over {ITERATIONS} iterations...")
        
        for _ in range(ITERATIONS):
            # Calculate forces (errors) for all nodes
            node_corrections = {i: {'dx': 0, 'dy': 0, 'dt': 0, 'c': 0} for i in range(len(self.nodes))}
            land_corrections = {i: {'dx': 0, 'dy': 0, 'c': 0} for i in range(len(self.landmarks))}

            for c in self.constraints:
                # 1. Odometry Springs (Robot -> Robot)
                if c['type'] == 'odometry':
                    n1 = self.nodes[c['i']]
                    n2 = self.nodes[c['j']]
                    
                    # Expected position of n2 based on n1
                    pred_x = n1['x'] + c['dx']
                    pred_y = n1['y'] + c['dy']
                    
                    # Error
                    ex = pred_x - n2['x']
                    ey = pred_y - n2['y']
                    
                    # Distribute correction (pull both nodes)
                    if not n1['fixed']:
                        node_corrections[c['i']]['dx'] += ex * 0.5
                        node_corrections[c['i']]['dy'] += ey * 0.5
                        node_corrections[c['i']]['c'] += 1
                    
                    if not n2['fixed']:
                        node_corrections[c['j']]['dx'] -= ex * 0.5
                        node_corrections[c['j']]['dy'] -= ey * 0.5
                        node_corrections[c['j']]['c'] += 1

                # 2. Measurement Springs (Robot <-> Landmark)
                elif c['type'] == 'measurement':
                    robot = self.nodes[c['node_idx']]
                    land = self.landmarks[c['land_idx']]
                    
                    # Where the landmark SHOULD be relative to robot
                    angle = robot['theta'] + c['angle_offset']
                    pred_lx = robot['x'] + c['dist'] * math.cos(angle)
                    pred_ly = robot['y'] + c['dist'] * math.sin(angle)
                    
                    # Error vector
                    ex = pred_lx - land['x']
                    ey = pred_ly - land['y']
                    
                    # Pull Robot towards consistency
                    if not robot['fixed']:
                        node_corrections[c['node_idx']]['dx'] -= ex * 0.5
                        node_corrections[c['node_idx']]['dy'] -= ey * 0.5
                        node_corrections[c['node_idx']]['c'] += 1
                        
                    # Pull Landmark towards consistency
                    land_corrections[c['land_idx']]['dx'] += ex * 0.5
                    land_corrections[c['land_idx']]['dy'] += ey * 0.5
                    land_corrections[c['land_idx']]['c'] += 1

            # Apply Corrections
            for i, corr in node_corrections.items():
                if corr['c'] > 0 and not self.nodes[i]['fixed']:
                    self.nodes[i]['x'] += (corr['dx'] / corr['c']) * LEARNING_RATE
                    self.nodes[i]['y'] += (corr['dy'] / corr['c']) * LEARNING_RATE

            for i, corr in land_corrections.items():
                if corr['c'] > 0:
                    self.landmarks[i]['x'] += (corr['dx'] / corr['c']) * LEARNING_RATE
                    self.landmarks[i]['y'] += (corr['dy'] / corr['c']) * LEARNING_RATE

# --- 3. Main Logic ---

def process_map_data(x_raw, y_raw, theta_raw, distance_measured):
    
    # Initialize SLAM System
    slam = GraphSLAM()
    
    # 1. Build the Graph
    print("Building Pose Graph...")
    for i in range(len(x_raw)):
        # Add Robot Node
        slam.add_robot_node(x_raw[i], y_raw[i], theta_raw[i])
        
        # Add Measurement (if valid)
        if distance_measured[i] < 200: # Ignore max-range readings
            slam.add_measurement(i, distance_measured[i])

    # 2. Optimize (Relax Springs)
    slam.relax_springs()
    
    # 3. Extract Optimized Data
    opt_rx = [n['x'] for n in slam.nodes]
    opt_ry = [n['y'] for n in slam.nodes]
    
    opt_lx = [l['x'] for l in slam.landmarks]
    opt_ly = [l['y'] for l in slam.landmarks]
    
    # --- Plotting ---
    plt.figure(figsize=(12, 10))
    
    # Plot Raw Odometry (Faint Red)
    plt.plot(x_raw, y_raw, 'r--', alpha=0.3, label='Raw Odometry')
    
    # Plot Optimized Path (Blue)
    plt.plot(opt_rx, opt_ry, 'b-', linewidth=2, label='Optimized Path (Springs)')
    
    # Plot Landmarks (Green dots)
    plt.scatter(opt_lx, opt_ly, c='g', s=10, alpha=0.6, label='Corrected Walls')
    
    plt.title('Graph SLAM: Spring-Based Correction', fontsize=16)
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