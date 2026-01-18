import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to your image
IMAGE_PATH = "./images/camera_snapshot.png"

def order_points(pts):
    """
    Orders coordinates in the order: top-left, top-right, bottom-right, bottom-left.
    This is critical for the perspective transform to work correctly.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def get_arena_geometry(image_path):
    # 1. Load and Preprocess
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to smooth out the concrete texture and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    # Thresholds are tuned for the high contrast between wood and floor
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate the edges to close any small gaps in the contour
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 2. Find Contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the arena is the largest contour in the image
    arena_contour = max(contours, key=cv2.contourArea)
    
    # 3. Simplify Contour to find Corners
    # We use Douglas-Peucker algorithm (approxPolyDP) to simplify the shape
    epsilon = 0.02 * cv2.arcLength(arena_contour, True)
    approx_corners = cv2.approxPolyDP(arena_contour, epsilon, True)
    
    # Check if we found 4 corners (for a rectangular arena)
    # If not, we take the Convex Hull to force a convex shape
    if len(approx_corners) != 4:
        print(f"Detected {len(approx_corners)} sides. Using Convex Hull to standardize.")
        hull = cv2.convexHull(approx_corners)
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx_corners = cv2.approxPolyDP(hull, epsilon, True)

    # Ensure we have exactly 4 points for Homography (Perspective Transform)
    # If approxPolyDP failed to find 4, we assume the bounding box or logic needs manual tuning.
    # For this specific image, the frame is very distinct, so approx usually works.
    if len(approx_corners) != 4:
        print("Warning: Could not simplify to 4 distinct corners. Using Bounding Rect.")
        rect = cv2.minAreaRect(arena_contour)
        box = cv2.boxPoints(rect)
        approx_corners = np.int0(box)

    # Reshape for the order_points function
    pts = approx_corners.reshape(4, 2)
    ordered_rect = order_points(pts)

    # 4. Perspective Transform (Homography)
    # Define the dimensions of the new "top-down" view (e.g., 600x800 px)
    # You can calculate max width/height for accuracy, but fixed is fine for mapping
    (tl, tr, br, bl) = ordered_rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate the transformation matrix M
    M = cv2.getPerspectiveTransform(ordered_rect, dst)
    
    # Warp the image (Optional, just to check visual correctness)
    warped_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return ordered_rect, dst, M, (maxWidth, maxHeight)

def plot_geometry_map(dst_dims):
    maxWidth, maxHeight = dst_dims
    
    # 5. Create the Geometry Map Plot
    # We define the arena boundaries based on the warped dimensions
    # Since we warped the arena to be the full image size, the walls are the image borders.
    
    # Define coordinates for the 4 corners of the map
    map_corners = [
        (0, 0),
        (maxWidth, 0),
        (maxWidth, maxHeight),
        (0, maxHeight),
        (0, 0) # Close the loop
    ]
    
    x_vals = [p[0] for p in map_corners]
    y_vals = [p[1] for p in map_corners]

    plt.figure(figsize=(6, 8))
    
    # Plot the Arena Walls
    plt.plot(x_vals, y_vals, marker='o', color='blue', linewidth=3, label='Arena Walls')
    
    # Plot "Virtual" Grid (helpful for robot tracking)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Settings for pure geometry view
    plt.title("Generated 2D Arena Geometry")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    
    # Invert Y axis because image coordinates start from top-left, 
    # but typically robot maps start bottom-left.
    plt.gca().invert_yaxis() 
    
    plt.show()

if __name__ == "__main__":
    try:
        # Detect and Transform
        src_rect, dst_rect, Matrix, dims = get_arena_geometry(IMAGE_PATH)
        
        print("Arena Corners Detected (Source Image):")
        print(src_rect)
        print("\nHomography Matrix Calculated.")
        
        # Plot only the geometry
        plot_geometry_map(dims)
        
    except Exception as e:
        print(f"Error: {e}")