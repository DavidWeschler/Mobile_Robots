import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration matching the NXC code ---
FILE_NAME = "EX2/plots/MAP.TXT"

# Scaling factors used in the NXC code (to convert float to long/int)
X_Y_SCALE = 10000.0  # x_cm and y_cm were multiplied by 10000
THETA_SCALE = 100000.0 # theta_rad was multiplied by 100000

# --- Sensor Placement Assumption ---
# Wall-following usually means the US sensor is mounted PERPENDICULAR to the robot's
# direction of travel (e.g., on the right side).
# We assume the robot is driving *parallel* to the wall and measuring distance D 
# to its right. The robot's heading is theta.
# The wall point vector is calculated by rotating the distance D 90 degrees 
# CLOCKWISE (or -pi/2 radians) from the robot's current heading.
SENSOR_ANGLE_OFFSET = (math.pi / 2.0) # -90 degrees or -1.57 radians (for right wall)

def calculate_wall_coordinates(x, y, theta, distance):
    """
    Calculates the absolute (x, y) coordinates of the wall point based on the 
    robot's pose and the ultrasonic reading.
    """
    # 1. Calculate the sensor's absolute heading
    sensor_heading = theta + SENSOR_ANGLE_OFFSET
    
    # 2. Project the distance D along the sensor's heading vector
    wall_x = x + distance * math.cos(sensor_heading)
    wall_y = y + distance * math.sin(sensor_heading)
    
    return wall_x, wall_y

def load_and_plot_data():
    """
    Loads data from MAP.TXT, calculates wall points, and generates the plot.
    """
    try:
        # Load the raw data from the file
        data = np.loadtxt(FILE_NAME, delimiter=',')
    except FileNotFoundError:
        print(f"Error: File '{FILE_NAME}' not found. Make sure you transfer it from the NXT brick.")
        return
    except Exception as e:
        print(f"An error occurred while loading or parsing the data: {e}")
        print("Please check the format of your MAP.TXT file.")
        return

    # Separate the columns (assuming 4 columns: xo, yo, to, D)
    xo_raw = data[:, 0]
    yo_raw = data[:, 1]
    to_raw = data[:, 2]
    D_raw = data[:, 3]

    # --- Rescale Data back to cm and radians ---
    # The NXC code scaled these by multiplying. We divide here.
    x_robot = xo_raw / X_Y_SCALE
    y_robot = yo_raw / X_Y_SCALE
    theta_robot = to_raw / THETA_SCALE
    
    # D_raw is already in cm (integer)
    distance_measured = D_raw
    
    # --- Calculate Wall Coordinates (Mapping) ---
    x_wall_map = []
    y_wall_map = []
    
    for i in range(len(x_robot)):
        # Calculate wall position for each logged point
        wx, wy = calculate_wall_coordinates(
            x_robot[i], 
            y_robot[i], 
            theta_robot[i], 
            distance_measured[i]
        )
        x_wall_map.append(wx)
        y_wall_map.append(wy)
        
    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # 1. Plot the Robot's Path (Odometry)
    plt.plot(x_robot, y_robot, label='Robot Path (Odometry)', 
             color='#1f77b4', linewidth=2, linestyle='--')
    plt.plot(x_robot[0], y_robot[0], 'go', markersize=8, label='Start Position')
    plt.plot(x_robot[-1], y_robot[-1], 'ro', markersize=8, label='End Position')

    # 2. Plot the Detected Wall (Sensor Data)
    # Using 'o' markers helps show individual sensor readings.
    plt.plot(x_wall_map, y_wall_map, 'o', label='Detected Wall Points', 
             color='#ff7f0e', markersize=3, alpha=0.6)
    
    # Set plot details
    plt.title('Robot Wall Following Path and Map', fontsize=16)
    plt.xlabel('X Position (cm)', fontsize=12)
    plt.ylabel('Y Position (cm)', fontsize=12)
    
    # Ensure equal scaling on both axes so the turns look correct
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    load_and_plot_data()