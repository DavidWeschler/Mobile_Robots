import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import glob
import os


# =============================================================================
# FILE PATHS
# =============================================================================
CORNERS_FILE_PATH = "txt_files/clean_walls.TXT"
ODOMETRY_TRACK_FILE = "txt_files/robot_path.TXT"  # Robot's odometry trajectory
CAMERA_TRACK_FILE = "txt_files/trajectory.txt"

# Updated to 5 vertices to match measured data
true_vertices = np.array([
    [0, 0],       # Top-Left
    [190, 0],     # Top-Right
    [190, -237],   # Bottom-Right
    [89, -237],    # Bottom-Inner
    [0, -148]      # Left-Inner
])


def load_vertices(path, tolerance=20.0, min_wall_length=50.0):
    """
    Load vertices from wall segments CSV file.
    Format: X_Start,Y_Start,X_End,Y_End,Length_cm (with header)
    Extracts unique endpoints by clustering nearby points, then orders by angle.
    
    Args:
        path: Path to CSV file
        tolerance: Distance threshold for clustering endpoints (cm)
        min_wall_length: Minimum wall length to include (filters short segments)
    """
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    # Columns: X_Start, Y_Start, X_End, Y_End, Length_cm
    
    # Filter walls by minimum length
    if data.shape[1] >= 5:  # Has length column
        lengths = data[:, 4]
        data = data[lengths >= min_wall_length]
    
    starts = data[:, 0:2]  # (N, 2)
    ends = data[:, 2:4]    # (N, 2)
    
    # Collect all endpoints
    all_points = np.vstack([starts, ends])
    
    # Cluster nearby points to find unique vertices
    vertices = []
    used = [False] * len(all_points)
    
    for i, pt in enumerate(all_points):
        if used[i]:
            continue
        # Find all points within tolerance of this point
        cluster = [pt]
        used[i] = True
        for j in range(i + 1, len(all_points)):
            if not used[j] and np.linalg.norm(all_points[j] - pt) < tolerance:
                cluster.append(all_points[j])
                used[j] = True
        # Average the cluster to get the vertex
        vertices.append(np.mean(cluster, axis=0))
    
    vertices = np.array(vertices)
    
    # Order vertices by angle from centroid (counter-clockwise)
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    order = np.argsort(angles)
    vertices = vertices[order]
    
    return vertices


def load_odometry_trajectory(path):
    """
    Load robot's odometry trajectory from robot_path.txt file.
    Format: X_cm,Y_cm (with header)
    Returns: numpy array of (x, y) in cm
    """
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    # Data is already in cm, columns 0 and 1
    trajectory = data[:, 0:2]
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


def close_polygon(vertices):
    """Close a polygon by appending the first vertex at the end."""
    return np.vstack([vertices, vertices[0]])


def find_best_vertices_alignment(measured, true_ref):
    """
    Align measured vertices to true reference using translation, rotation, and optional mirroring.
    Uses Procrustes-style alignment with vertex matching.
    Returns: (aligned_vertices, transform_params, error)
    where transform_params is a dict with mirror_x, mirror_y, rotation_matrix, measured_centroid, true_centroid
    """
    from scipy.spatial.distance import cdist
    
    best_error = float('inf')
    best_result = None
    best_transform = None
    
    # Center both on their centroids for comparison
    true_centroid = np.mean(true_ref, axis=0)
    true_centered = true_ref - true_centroid
    
    for mirror_x in [False, True]:
        for mirror_y in [False, True]:
            # Apply mirroring to measured
            v = measured.copy()
            if mirror_x:
                v[:, 0] = -v[:, 0]
            if mirror_y:
                v[:, 1] = -v[:, 1]
            
            # Center measured
            measured_centroid = np.mean(v, axis=0)
            v_centered = v - measured_centroid
            
            # Try different starting vertex alignments (cyclic rotations of vertex order)
            n_meas = len(v_centered)
            n_true = len(true_centered)
            
            for start_idx in range(n_meas):
                # Rotate vertex order
                v_rotated_order = np.roll(v_centered, -start_idx, axis=0)
                
                # Find optimal rotation angle to align first vertex direction
                if n_meas > 0 and n_true > 0:
                    # Use first vertex to estimate rotation
                    angle_true = np.arctan2(true_centered[0, 1], true_centered[0, 0])
                    angle_meas = np.arctan2(v_rotated_order[0, 1], v_rotated_order[0, 0])
                    rotation_angle = angle_true - angle_meas
                    
                    # Apply rotation
                    cos_a = np.cos(rotation_angle)
                    sin_a = np.sin(rotation_angle)
                    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    v_aligned = v_rotated_order @ rotation_matrix.T
                    
                    # Compute error as sum of nearest-neighbor distances
                    distances = cdist(v_aligned, true_centered)
                    # Use Hungarian-style matching for best correspondence
                    nn_dist = np.min(distances, axis=1)
                    error = np.mean(nn_dist)
                    
                    if error < best_error:
                        best_error = error
                        # Apply same transformations to original (not centered) for final result
                        v_final = measured.copy()
                        if mirror_x:
                            v_final[:, 0] = -v_final[:, 0]
                        if mirror_y:
                            v_final[:, 1] = -v_final[:, 1]
                        v_final = np.roll(v_final, -start_idx, axis=0)
                        
                        # Recompute centroid after mirroring for transform storage
                        mirrored_centroid = np.mean(v_final, axis=0)
                        v_final_centered = v_final - mirrored_centroid
                        v_final_rotated = v_final_centered @ rotation_matrix.T
                        # Translate to match true centroid
                        v_final_aligned = v_final_rotated + true_centroid
                        
                        # Store transform parameters
                        best_transform = {
                            'mirror_x': mirror_x,
                            'mirror_y': mirror_y,
                            'rotation_matrix': rotation_matrix,
                            'rotation_deg': np.degrees(rotation_angle),
                            'measured_centroid': mirrored_centroid,
                            'true_centroid': true_centroid
                        }
                        best_result = (v_final_aligned, best_transform, best_error)
    
    return best_result


def apply_arena_transform(points, transform_params):
    """
    Apply the same transformation used for arena alignment to a set of points.
    This ensures odometry trajectory is in the same coordinate system as the aligned arena.
    """
    pts = points.copy()
    
    # Step 1: Apply mirroring
    if transform_params['mirror_x']:
        pts[:, 0] = -pts[:, 0]
    if transform_params['mirror_y']:
        pts[:, 1] = -pts[:, 1]
    
    # Step 2: Center (subtract measured centroid)
    pts_centered = pts - transform_params['measured_centroid']
    
    # Step 3: Rotate
    pts_rotated = pts_centered @ transform_params['rotation_matrix'].T
    
    # Step 4: Translate to true coordinate system
    pts_transformed = pts_rotated + transform_params['true_centroid']
    
    return pts_transformed


# =============================================================================
# PART 1: CORNER COMPARISON
# =============================================================================
print("=" * 60)
print("       PART 1: ARENA CORNER COMPARISON")
print("=" * 60)

measured_vertices_raw = load_vertices(CORNERS_FILE_PATH)

# Find best alignment for corners (mirror + rotation + translation)
measured_vertices, arena_transform, align_error = find_best_vertices_alignment(measured_vertices_raw, true_vertices)

mirror_str = []
if arena_transform['mirror_x']:
    mirror_str.append("X")
if arena_transform['mirror_y']:
    mirror_str.append("Y")
alignment_info = []
if mirror_str:
    alignment_info.append(f"mirror: {', '.join(mirror_str)}")
alignment_info.append(f"rotation: {arena_transform['rotation_deg']:.1f}°")
print(f"   Alignment: {', '.join(alignment_info)} (error: {align_error:.1f} cm)")

print(f"\nTrue vertices: {len(true_vertices)}, Measured vertices: {len(measured_vertices)}")

true_lengths = compute_edge_lengths(true_vertices)
measured_lengths = compute_edge_lengths(measured_vertices)

true_angles = compute_angles(true_vertices)
measured_angles = compute_angles(measured_vertices)

# Handle mismatched vertex counts by comparing minimum common length
n_compare = min(len(true_lengths), len(measured_lengths))
if len(true_lengths) != len(measured_lengths):
    print(f"⚠️ Vertex count mismatch! Comparing first {n_compare} edges/angles.")

length_error_abs = mean_absolute_error(measured_lengths[:n_compare], true_lengths[:n_compare])
length_error_percent = (length_error_abs / np.mean(true_lengths[:n_compare])) * 100
angle_error_abs = mean_absolute_error(measured_angles[:n_compare], true_angles[:n_compare])
angle_error_percent = (angle_error_abs / np.mean(true_angles[:n_compare])) * 100

print("\n=== Corner Accuracy Results ===")
print(f"Mean Edge Length Error: {length_error_abs:.2f} cm ({length_error_percent:.2f}%)")
print(f"Mean Angle Error: {angle_error_abs:.2f}° ({angle_error_percent:.2f}%)")


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
    
    # Transform odometry trajectory using the same transform as the arena
    # This places the odometry in the same coordinate system as the aligned arena
    odometry_transformed = apply_arena_transform(odometry_traj, arena_transform)
    
    # Use transformed odometry as the aligned version (it stays in place)
    odometry_aligned = odometry_transformed
    
    # Camera trajectory is already in true arena coordinates - use as-is
    camera_aligned = camera_traj.copy()
    
    print(f"   Odometry start (transformed): ({odometry_aligned[0, 0]:.2f}, {odometry_aligned[0, 1]:.2f})")
    print(f"   Camera start: ({camera_aligned[0, 0]:.2f}, {camera_aligned[0, 1]:.2f})")
    
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

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Color scheme
TRUE_COLOR = '#2ecc71'  # Green
MEASURED_COLOR = '#e74c3c'  # Red
ODOM_COLOR = '#3498db'  # Blue
CAM_COLOR = '#e67e22'  # Orange

# =============================================================================
# FIRST FIGURE: Combined Arena + Trajectories View
# =============================================================================
print("\n" + "=" * 60)
print("       COMBINED VIEW")
print("=" * 60)

fig2, ax2 = plt.subplots(figsize=(12, 10))

# Plot arenas
closed_true = close_polygon(true_vertices)
closed_measured = close_polygon(measured_vertices)

ax2.fill(closed_true[:, 0], closed_true[:, 1], alpha=0.15, color=TRUE_COLOR)
ax2.fill(closed_measured[:, 0], closed_measured[:, 1], alpha=0.15, color=MEASURED_COLOR)
ax2.plot(closed_true[:, 0], closed_true[:, 1], '-', color=TRUE_COLOR, linewidth=3, marker='o', markersize=10, label='True Arena')
ax2.plot(closed_measured[:, 0], closed_measured[:, 1], '--', color=MEASURED_COLOR, linewidth=3, marker='s', markersize=9, label='Measured Arena (SLAM)')

# Add vertex labels for arenas
for i, v in enumerate(true_vertices):
    ax2.annotate(f'T{i}', (v[0], v[1]), textcoords="offset points", xytext=(8, 8), fontsize=10, color=TRUE_COLOR, fontweight='bold')
for i, v in enumerate(measured_vertices):
    ax2.annotate(f'M{i}', (v[0], v[1]), textcoords="offset points", xytext=(-15, -15), fontsize=10, color=MEASURED_COLOR, fontweight='bold')

# Plot trajectories if available
if trajectory_comparison_available:
    ax2.plot(odometry_aligned[:, 0], odometry_aligned[:, 1], '-', color=ODOM_COLOR, linewidth=2, alpha=0.8, label='Odometry Trajectory')
    ax2.plot(camera_aligned[:, 0], camera_aligned[:, 1], '-', color=CAM_COLOR, linewidth=2, alpha=0.8, label='Camera Trajectory')
    
    # Start markers at actual trajectory starts
    ax2.scatter(odometry_aligned[0, 0], odometry_aligned[0, 1], c='#9b59b6', s=250, marker='*', zorder=10, edgecolors='white', linewidth=2, label='Trajectory Start')
    
    # End markers
    ax2.scatter(odometry_aligned[-1, 0], odometry_aligned[-1, 1], c=ODOM_COLOR, s=120, marker='X', zorder=10, edgecolors='white', linewidth=2)
    ax2.scatter(camera_aligned[-1, 0], camera_aligned[-1, 1], c=CAM_COLOR, s=120, marker='X', zorder=10, edgecolors='white', linewidth=2)

ax2.set_xlabel('X [cm]', fontsize=12)
ax2.set_ylabel('Y [cm]', fontsize=12)
ax2.set_title('Combined View: Arena & Trajectories', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.0, 1.0))
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

print("\n✅ Combined view generated. Close the window to see summary plots.")
plt.show()

# =============================================================================
# SECOND FIGURE: Detailed Comparison with Subplots
# =============================================================================

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_corners = axes[0, 0]
ax_traj = axes[1, 0]
ax_error = axes[0, 1]
ax_stats = axes[1, 1]

# --- Plot 1: Arena Comparison ---
closed_true = close_polygon(true_vertices)
closed_measured = close_polygon(measured_vertices)
ax_corners.fill(closed_true[:, 0], closed_true[:, 1], alpha=0.2, color=TRUE_COLOR)
ax_corners.fill(closed_measured[:, 0], closed_measured[:, 1], alpha=0.2, color=MEASURED_COLOR)
ax_corners.plot(closed_true[:, 0], closed_true[:, 1], '-', color=TRUE_COLOR, linewidth=2.5, marker='o', markersize=8, label='True Arena')
ax_corners.plot(closed_measured[:, 0], closed_measured[:, 1], '--', color=MEASURED_COLOR, linewidth=2.5, marker='s', markersize=7, label='Measured (SLAM)')

# Add vertex labels
for i, v in enumerate(true_vertices):
    ax_corners.annotate(f'T{i}', (v[0], v[1]), textcoords="offset points", xytext=(5, 5), fontsize=9, color=TRUE_COLOR, fontweight='bold')
for i, v in enumerate(measured_vertices):
    ax_corners.annotate(f'M{i}', (v[0], v[1]), textcoords="offset points", xytext=(-12, -12), fontsize=9, color=MEASURED_COLOR, fontweight='bold')

ax_corners.set_xlabel('X [cm]')
ax_corners.set_ylabel('Y [cm]')
ax_corners.set_title(f'Arena Shape Comparison\nLength Error: {length_error_percent:.1f}%, Angle Error: {angle_error_percent:.1f}%')
ax_corners.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
ax_corners.set_aspect('equal')
ax_corners.grid(True, alpha=0.3)

# --- Plot 2: Pointwise Error Distribution ---
if trajectory_comparison_available:
    n, bins, patches = ax_error.hist(point_errors, bins=25, color=ODOM_COLOR, edgecolor='white', alpha=0.8)
    ax_error.axvline(traj_metrics['mean_error_cm'], color=MEASURED_COLOR, linestyle='-', linewidth=2.5, label=f'Mean: {traj_metrics["mean_error_cm"]:.1f} cm')
    ax_error.axvline(traj_metrics['median_error_cm'], color='#f39c12', linestyle='--', linewidth=2.5, label=f'Median: {traj_metrics["median_error_cm"]:.1f} cm')
    ax_error.set_xlabel('Error [cm]')
    ax_error.set_ylabel('Frequency')
    ax_error.set_title('Pointwise Error Distribution')
    ax_error.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax_error.grid(True, alpha=0.3, axis='y')
else:
    ax_error.text(0.5, 0.5, 'No trajectory data available', ha='center', va='center', fontsize=12, transform=ax_error.transAxes)
    ax_error.set_title('Pointwise Error Distribution')

# --- Plot 3: Aligned Trajectories ---
if trajectory_comparison_available:
    ax_traj.plot(odometry_aligned[:, 0], odometry_aligned[:, 1], '-', color=ODOM_COLOR, linewidth=2, alpha=0.9, label='Odometry')
    ax_traj.plot(camera_aligned[:, 0], camera_aligned[:, 1], '-', color=CAM_COLOR, linewidth=2, alpha=0.9, label='Camera')
    # Start marker at actual trajectory start
    ax_traj.scatter(odometry_aligned[0, 0], odometry_aligned[0, 1], c=TRUE_COLOR, s=200, marker='*', zorder=5, edgecolors='white', linewidth=2, label='Odom Start')
    ax_traj.scatter(camera_aligned[0, 0], camera_aligned[0, 1], c='#9b59b6', s=150, marker='*', zorder=5, edgecolors='white', linewidth=2, label='Cam Start')
    # End markers
    ax_traj.scatter(odometry_aligned[-1, 0], odometry_aligned[-1, 1], c=ODOM_COLOR, s=100, marker='X', zorder=5, edgecolors='white', linewidth=1)
    ax_traj.scatter(camera_aligned[-1, 0], camera_aligned[-1, 1], c=CAM_COLOR, s=100, marker='X', zorder=5, edgecolors='white', linewidth=1)
    ax_traj.set_xlabel('X [cm]')
    ax_traj.set_ylabel('Y [cm]')
    ax_traj.set_title(f'Aligned Trajectories\nRMSE: {traj_metrics["rmse_cm"]:.2f} cm')
    ax_traj.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)
else:
    ax_traj.text(0.5, 0.5, 'No trajectory data available', ha='center', va='center', fontsize=12, transform=ax_traj.transAxes)
    ax_traj.set_title('Aligned Trajectories')

# --- Plot 4: Summary Table ---
ax_stats.axis('off')

if trajectory_comparison_available:
    summary_text = """
┌───────────────────────────────────────────────────────────┐
│                   COMPARISON SUMMARY                      │
├───────────────────────────────────────────────────────────┤
│  ARENA CORNERS                                            │
│    True Vertices:       {n_true:>3}                               │
│    Measured Vertices:   {n_meas:>3}                               │
│    Edge Length Error:   {len_err:>6.2f} cm  ({len_pct:>5.1f}%)               │
│    Angle Error:         {ang_err:>6.2f}°   ({ang_pct:>5.1f}%)                │
├───────────────────────────────────────────────────────────┤
│  TRAJECTORY                                               │
│    Mean Error:          {mean_err:>6.2f} cm                         │
│    Median Error:        {med_err:>6.2f} cm                         │
│    Max Error:           {max_err:>6.2f} cm                         │
│    Std Dev:             {std_err:>6.2f} cm                         │
│    RMSE:                {rmse:>6.2f} cm                         │
└───────────────────────────────────────────────────────────┘
""".format(
        n_true=len(true_vertices), n_meas=len(measured_vertices),
        len_err=length_error_abs, len_pct=length_error_percent,
        ang_err=angle_error_abs, ang_pct=angle_error_percent,
        n_odom=len(odometry_traj), n_cam=len(camera_traj),
        mean_err=traj_metrics['mean_error_cm'],
        med_err=traj_metrics['median_error_cm'],
        max_err=traj_metrics['max_error_cm'],
        std_err=traj_metrics['std_error_cm'],
        rmse=traj_metrics['rmse_cm']
    )
else:
    summary_text = """
┌───────────────────────────────────────────────────────────┐
│                   COMPARISON SUMMARY                      │
├───────────────────────────────────────────────────────────┤
│  ARENA CORNERS                                            │
│  ───────────────────────────────────────────────────────  │
│    True Vertices:       {n_true:>3}                                │
│    Measured Vertices:   {n_meas:>3}                                │
│    Edge Length Error:   {len_err:>6.2f} cm  ({len_pct:>5.1f}%)              │
│    Angle Error:         {ang_err:>6.2f}°   ({ang_pct:>5.1f}%)              │
├───────────────────────────────────────────────────────────┤
│  TRAJECTORY                                               │
│  ───────────────────────────────────────────────────────  │
│    No trajectory data available                           │
└───────────────────────────────────────────────────────────┘
""".format(
        n_true=len(true_vertices), n_meas=len(measured_vertices),
        len_err=length_error_abs, len_pct=length_error_percent,
        ang_err=angle_error_abs, ang_pct=angle_error_percent
    )

ax_stats.text(0.5, 0.5, summary_text, transform=ax_stats.transAxes,
              fontsize=11, fontfamily='monospace', verticalalignment='center',
              horizontalalignment='center',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.9))

# Add overall title
fig.suptitle('SLAM Accuracy Analysis', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

print("\n✅ Summary plots generated. Close the window to exit.")
plt.show()

