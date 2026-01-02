import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = "plots/CENTER.TXT"

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


measured_vertices = load_vertices(FILE_PATH)
measured_vertices = rotate_to_align_base(measured_vertices)


true_lengths = compute_edge_lengths(true_vertices)
measured_lengths = compute_edge_lengths(measured_vertices)

true_angles = compute_angles(true_vertices)
measured_angles = compute_angles(measured_vertices)

length_error_abs = mean_absolute_error(measured_lengths, true_lengths)
length_error_percent = (length_error_abs / np.mean(true_lengths)) * 100
angle_error_abs = mean_absolute_error(measured_angles, true_angles)
angle_error_percent = (angle_error_abs / np.mean(true_angles)) * 100

print("=== Accuracy Results ===")
print(f"Mean Edge Length Error: {length_error_percent:.2f}%")
print(f"Mean Angle Error: {angle_error_percent:.2f}%")


def plot_polygon(vertices, label, style):
    closed = np.vstack([vertices, vertices[0]])
    plt.plot(closed[:, 0], closed[:, 1], style, label=label)

plt.figure(figsize=(8, 8))
plot_polygon(true_vertices, "True Shape", "g-o")
plot_polygon(measured_vertices, "Measured Shape", "r--o")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("True Shape vs Measured Shape")
plt.suptitle(f"Mean Edge Length Error: {length_error_percent:.2f}% | Mean Angle Error: {angle_error_percent:.2f}%", 
             fontsize=10, y=0.96)
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")
plt.show()