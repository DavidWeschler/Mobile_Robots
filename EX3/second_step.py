import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import glob
import os


# =============================================================================
# FILE PATHS
# =============================================================================
CORNERS_FILE_PATH = "txt_files/odometry_corners.TXT"
ODOMETRY_TRACK_FILE = "txt_files/TRACK_LOG.TXT"  # Robot's odometry trajectory
CAMERA_TRACK_FILE = "txt_files/trajectory.txt"

# Conversion factor: TRACK_LOG units to cm (adjust based on your robot's odometry)
# If odometry is in 0.01mm, divide by 100 to get mm, then by 10 to get cm = /1000
ODOMETRY_SCALE = 1000.0  # Divide odometry values by this to get cm

# Updated to 5 vertices to match measured data
true_vertices = np.array([
    [-41.5, -30],
    [-130.5, 59],
    [-130.5, 207],
    [59.5, 207],
    [59.5, -30]
])


def load_vertices(path):
    """Load vertices from file where first line is count, then alternating x,y coordinates."""
    data = np.loadtxt(path)
    n = int(data[0])  # First value is the number of vertices
    coords = data[1:]  # Remaining values are coordinates
    # Reshape into (n, 2) array
    vertices = coords.reshape(n, 2)
    return vertices


def load_odometry_trajectory(path, scale=ODOMETRY_SCALE):
    """
    Load robot's odometry trajectory from TRACK_LOG file.
    Format: timestamp,x,y,distance,flag
    Returns: numpy array of (x, y) in cm
    """
    data = np.loadtxt(path, delimiter=',')
    # Extract x, y columns (indices 1 and 2)
    trajectory = data[:, 1:3] / scale  # Convert to cm
    return trajectory


def load_camera_trajectory(path):
    """
    Load camera-tracked trajectory from trajectory_*.txt file.
    Format: x_cm,y_cm (with header lines starting with #)
    Returns: numpy array of (x, y) in cm
    """
    points = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                points.append([x, y])
    return np.array(points)

def align_trajectories(traj1, traj2):
    """
    Align two trajectories by translating traj2 so both start at the same point.
    Also handles rotation alignment using the first few points.
    """
    # Translate both to start at origin
    traj1_aligned = traj1 - traj1[0]
    traj2_aligned = traj2 - traj2[0]
    
    # Optionally rotate traj2 to align initial direction with traj1
    # Use first N points to determine initial direction
    n_points = min(10, len(traj1_aligned) - 1, len(traj2_aligned) - 1)
    if n_points > 1:
        # Calculate initial direction vectors
        dir1 = traj1_aligned[n_points] - traj1_aligned[0]
        dir2 = traj2_aligned[n_points] - traj2_aligned[0]
        
        # Calculate rotation angle
        angle1 = np.arctan2(dir1[1], dir1[0])
        angle2 = np.arctan2(dir2[1], dir2[0])
        rotation_angle = angle1 - angle2
        
        # Apply rotation to traj2
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        traj2_aligned = traj2_aligned @ rotation_matrix.T
    
    return traj1_aligned, traj2_aligned


def resample_trajectory(trajectory, n_points):
    """Resample trajectory to have exactly n_points using interpolation."""
    # Calculate cumulative distance along trajectory
    diffs = np.diff(trajectory, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # Create interpolation functions
    total_length = cumulative_dist[-1]
    if total_length == 0:
        return trajectory[:n_points] if len(trajectory) >= n_points else trajectory
    
    # New sample points evenly spaced
    new_dists = np.linspace(0, total_length, n_points)
    
    # Interpolate x and y
    interp_x = interp1d(cumulative_dist, trajectory[:, 0], kind='linear')
    interp_y = interp1d(cumulative_dist, trajectory[:, 1], kind='linear')
    
    resampled = np.column_stack([interp_x(new_dists), interp_y(new_dists)])
    return resampled


def compute_trajectory_errors(traj1, traj2):
    """
    Compute error metrics between two trajectories.
    Uses nearest-neighbor distance for each point.
    """
    # For each point in traj1, find distance to nearest point in traj2
    distances = cdist(traj1, traj2)
    
    # Nearest neighbor distances
    nn_dist_1_to_2 = np.min(distances, axis=1)
    nn_dist_2_to_1 = np.min(distances, axis=0)
    
    # Combine both directions
    all_nn_distances = np.concatenate([nn_dist_1_to_2, nn_dist_2_to_1])
    
    metrics = {
        'mean_error_cm': np.mean(all_nn_distances),
        'max_error_cm': np.max(all_nn_distances),
        'std_error_cm': np.std(all_nn_distances),
        'median_error_cm': np.median(all_nn_distances),
        'rmse_cm': np.sqrt(np.mean(all_nn_distances**2))
    }
    
    return metrics, nn_dist_1_to_2

def compute_edge_lengths(vertices):
    return np.linalg.norm(
        np.roll(vertices, -1, axis=0) - vertices,
        axis=1
    )

def compute_angles(vertices):
    angles = []
    n = len(vertices)

    for i in range(n):
        p_prev = vertices[i - 1]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        cos_theta = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        angles.append(angle)

    return np.array(angles)

def mean_absolute_error(a, b):
    return np.mean(np.abs(a - b))

def rotate_to_align_base(vertices):
    """Rotate vertices so the first edge (base) is parallel to x-axis."""
    # Calculate angle of first edge (from vertex 2 to vertex 3)
    edge = vertices[4] - vertices[3]
    angle = np.arctan2(edge[1], edge[0])
    
    # Create rotation matrix to rotate by -angle (to make horizontal)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Apply rotation around origin (0, 0)
    rotated = vertices @ rotation_matrix.T
    
    return rotated


# =============================================================================
# PART 1: CORNER COMPARISON
# =============================================================================
print("=" * 60)
print("       PART 1: ARENA CORNER COMPARISON")
print("=" * 60)

measured_vertices = load_vertices(CORNERS_FILE_PATH)
measured_vertices = rotate_to_align_base(measured_vertices)


true_lengths = compute_edge_lengths(true_vertices)
measured_lengths = compute_edge_lengths(measured_vertices)

true_angles = compute_angles(true_vertices)
measured_angles = compute_angles(measured_vertices)

length_error_abs = mean_absolute_error(measured_lengths, true_lengths)
length_error_percent = (length_error_abs / np.mean(true_lengths)) * 100
angle_error_abs = mean_absolute_error(measured_angles, true_angles)
angle_error_percent = (angle_error_abs / np.mean(true_angles)) * 100

print("\n=== Corner Accuracy Results ===")
print(f"Mean Edge Length Error: {length_error_abs:.2f} cm ({length_error_percent:.2f}%)")
print(f"Mean Angle Error: {angle_error_abs:.2f}° ({angle_error_percent:.2f}%)")


def plot_polygon(vertices, label, style):
    closed = np.vstack([vertices, vertices[0]])
    plt.plot(closed[:, 0], closed[:, 1], style, label=label)


# =============================================================================
# PART 2: TRAJECTORY COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("       PART 2: TRAJECTORY COMPARISON")
print("=" * 60)

# Auto-detect camera trajectory file if not specified
camera_file = CAMERA_TRACK_FILE

trajectory_comparison_available = False
odometry_traj = None
camera_traj = None
traj_metrics = None

# Load odometry trajectory
if os.path.exists(ODOMETRY_TRACK_FILE):
    print(f"\n📂 Loading odometry trajectory: {ODOMETRY_TRACK_FILE}")
    odometry_traj = load_odometry_trajectory(ODOMETRY_TRACK_FILE)
    print(f"   Loaded {len(odometry_traj)} points")
else:
    print(f"⚠️ Odometry trajectory file not found: {ODOMETRY_TRACK_FILE}")

# Load camera trajectory
if camera_file and os.path.exists(camera_file):
    print(f"📂 Loading camera trajectory: {camera_file}")
    camera_traj = load_camera_trajectory(camera_file)
    print(f"   Loaded {len(camera_traj)} points")
else:
    print("⚠️ Camera trajectory file not found.")
    print("   Run arena_tracker_system.py first to generate trajectory.txt")

# Compare trajectories if both available
if odometry_traj is not None and camera_traj is not None and len(odometry_traj) > 5 and len(camera_traj) > 5:
    trajectory_comparison_available = True
    
    # Align trajectories (translate to same start, rotate to align)
    odometry_aligned, camera_aligned = align_trajectories(odometry_traj, camera_traj)
    
    # Compute error metrics
    traj_metrics, point_errors = compute_trajectory_errors(odometry_aligned, camera_aligned)
    
    print("\n=== Trajectory Comparison Results ===")
    print(f"Mean Error:   {traj_metrics['mean_error_cm']:.2f} cm")
    print(f"Max Error:    {traj_metrics['max_error_cm']:.2f} cm")
    print(f"Median Error: {traj_metrics['median_error_cm']:.2f} cm")
    print(f"Std Error:    {traj_metrics['std_error_cm']:.2f} cm")
    print(f"RMSE:         {traj_metrics['rmse_cm']:.2f} cm")


# =============================================================================
# PLOTTING
# =============================================================================
print("\n" + "=" * 60)
print("       GENERATING PLOTS")
print("=" * 60)

# Create figure with subplots
if trajectory_comparison_available:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_corners = axes[0, 0]
    ax_traj_raw = axes[0, 1]
    ax_traj_aligned = axes[1, 0]
    ax_error = axes[1, 1]
else:
    fig, ax_corners = plt.subplots(1, 1, figsize=(8, 8))

# --- Plot 1: Corner Comparison ---
if trajectory_comparison_available:
    ax = ax_corners
else:
    ax = ax_corners

closed_true = np.vstack([true_vertices, true_vertices[0]])
closed_measured = np.vstack([measured_vertices, measured_vertices[0]])
ax.plot(closed_true[:, 0], closed_true[:, 1], 'g-o', linewidth=2, markersize=8, label='True Shape')
ax.plot(closed_measured[:, 0], closed_measured[:, 1], 'r--o', linewidth=2, markersize=8, label='Measured Shape (Odometry)')
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_title('Arena Corners: True vs Odometry')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

if trajectory_comparison_available:
    # --- Plot 2: Raw Trajectories ---
    ax_traj_raw.plot(odometry_traj[:, 0], odometry_traj[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Odometry')
    ax_traj_raw.plot(camera_traj[:, 0], camera_traj[:, 1], 'r-', linewidth=1.5, alpha=0.7, label='Camera')
    ax_traj_raw.scatter(odometry_traj[0, 0], odometry_traj[0, 1], c='blue', s=100, marker='o', zorder=5, label='Odom Start')
    ax_traj_raw.scatter(camera_traj[0, 0], camera_traj[0, 1], c='red', s=100, marker='s', zorder=5, label='Cam Start')
    ax_traj_raw.set_xlabel('X [cm]')
    ax_traj_raw.set_ylabel('Y [cm]')
    ax_traj_raw.set_title('Raw Trajectories (Before Alignment)')
    ax_traj_raw.legend()
    ax_traj_raw.grid(True, alpha=0.3)
    ax_traj_raw.axis('equal')
    
    # --- Plot 3: Aligned Trajectories ---
    ax_traj_aligned.plot(odometry_aligned[:, 0], odometry_aligned[:, 1], 'b-', linewidth=2, alpha=0.8, label='Odometry (Aligned)')
    ax_traj_aligned.plot(camera_aligned[:, 0], camera_aligned[:, 1], 'r-', linewidth=2, alpha=0.8, label='Camera (Aligned)')
    ax_traj_aligned.scatter(0, 0, c='green', s=150, marker='*', zorder=5, label='Common Start')
    ax_traj_aligned.scatter(odometry_aligned[-1, 0], odometry_aligned[-1, 1], c='blue', s=100, marker='x', zorder=5)
    ax_traj_aligned.scatter(camera_aligned[-1, 0], camera_aligned[-1, 1], c='red', s=100, marker='x', zorder=5)
    ax_traj_aligned.set_xlabel('X [cm]')
    ax_traj_aligned.set_ylabel('Y [cm]')
    ax_traj_aligned.set_title(f'Aligned Trajectories (RMSE: {traj_metrics["rmse_cm"]:.2f} cm)')
    ax_traj_aligned.legend()
    ax_traj_aligned.grid(True, alpha=0.3)
    ax_traj_aligned.axis('equal')
    
    # --- Plot 4: Error Distribution ---
    ax_error.hist(point_errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax_error.axvline(traj_metrics['mean_error_cm'], color='red', linestyle='--', linewidth=2, label=f'Mean: {traj_metrics["mean_error_cm"]:.2f} cm')
    ax_error.axvline(traj_metrics['median_error_cm'], color='orange', linestyle='--', linewidth=2, label=f'Median: {traj_metrics["median_error_cm"]:.2f} cm')
    ax_error.set_xlabel('Error [cm]')
    ax_error.set_ylabel('Frequency')
    ax_error.set_title('Trajectory Error Distribution')
    ax_error.legend()
    ax_error.grid(True, alpha=0.3)

plt.tight_layout()

# Add overall title
title_text = f"Corner Error: {length_error_percent:.1f}% (length), {angle_error_percent:.1f}% (angle)"
if trajectory_comparison_available:
    title_text += f" | Trajectory RMSE: {traj_metrics['rmse_cm']:.2f} cm"
plt.suptitle(title_text, fontsize=12, y=1.01)

print("\n✅ Plots generated. Close the window to exit.")
plt.show()