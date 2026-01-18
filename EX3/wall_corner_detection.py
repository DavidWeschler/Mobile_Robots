import cv2
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
IMAGE_PATH = "EX3\\images\\arena_logitech.png"
# IMAGE_PATH = "EX3\\images\\arena_empty.jpg"
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 100
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 10
ANGLE_TOLERANCE = 10  # pixels tolerance for horizontal/vertical

# ===============================
# HELPER FUNCTIONS
# ===============================

def line_intersection(l1, l2):
    """
    Compute intersection point of two lines.
    Each line is (x1, y1, x2, y2)
    Returns (x, y) or None if parallel
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    A = np.array([
        [x2 - x1, x3 - x4],
        [y2 - y1, y3 - y4]
    ])
    B = np.array([
        x3 - x1,
        y3 - y1
    ])

    if abs(np.linalg.det(A)) < 1e-6:
        return None

    t, _ = np.linalg.solve(A, B)
    xi = int(x1 + t * (x2 - x1))
    yi = int(y1 + t * (y2 - y1))
    return (xi, yi)


def remove_duplicates(points, min_dist=20):
    """
    Remove near-duplicate points
    """
    unique = []
    for p in points:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > min_dist for q in unique):
            unique.append(p)
    return unique


# ===============================
# MAIN PIPELINE
# ===============================

# Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

orig = img.copy()

# Grayscale + blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

# Hough line detection
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=HOUGH_THRESHOLD,
    minLineLength=MIN_LINE_LENGTH,
    maxLineGap=MAX_LINE_GAP
)

if lines is None:
    raise RuntimeError("No lines detected")

# Classify lines
horizontal_lines = []
vertical_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]

    if abs(y1 - y2) < ANGLE_TOLERANCE:
        horizontal_lines.append((x1, y1, x2, y2))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    elif abs(x1 - x2) < ANGLE_TOLERANCE:
        vertical_lines.append((x1, y1, x2, y2))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ===============================
# CORNER DETECTION
# ===============================

corners = []

for h in horizontal_lines:
    for v in vertical_lines:
        pt = line_intersection(h, v)
        if pt:
            corners.append(pt)

corners = remove_duplicates(corners)

# ===============================
# VISUALIZATION
# ===============================

for (x, y) in corners:
    cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

# ===============================
# OUTPUT RESULTS
# ===============================

num_walls = len(horizontal_lines) + len(vertical_lines)

print("===================================")
print("ARENA DETECTION RESULTS")
print("===================================")
print(f"Detected walls (line segments): {num_walls}")
print(f"Detected corners: {len(corners)}")
print("Corner pixel coordinates:")
for c in corners:
    print(c)

# ===============================
# DISPLAY WINDOWS
# ===============================

cv2.imshow("Edges", edges)
cv2.imshow("Walls & Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
