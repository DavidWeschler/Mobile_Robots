import cv2
import numpy as np
import matplotlib.pyplot as plt

# IMAGE_PATH = "./images/camera_snapshot_colored.png"
IMAGE_PATH = "./EX3/images/camera_snapshot_colored.png"

def get_red_mask(img_hsv):
    """ Isolates Red Tape """
    lower1, upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower2, upper2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    return cv2.inRange(img_hsv, lower1, upper1) + cv2.inRange(img_hsv, lower2, upper2)

def order_rect_points(pts):
    """ 
    Orders the 4 points of the VIRTUAL Bounding Box for Homography.
    (TL, TR, BR, BL)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def process_flexible_arena(image_path):
    # 1. Load & Preprocess
    img = cv2.imread(image_path)
    if img is None: raise Exception(f"Image not found: {image_path}")
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Get Red Mask
    mask = get_red_mask(hsv)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill gaps

    # 3. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise Exception("No Red Tape detected!")
    
    arena_contour = max(contours, key=cv2.contourArea)

    # 4. Flexible Corner Detection (N-Corners)
    # We do NOT use convexHull here, so we can detect concave shapes (like L-shapes)
    epsilon = 0.015 * cv2.arcLength(arena_contour, True) # 1.5% accuracy
    polygon_corners = cv2.approxPolyDP(arena_contour, epsilon, True)
    
    num_corners = len(polygon_corners)
    print(f"Detected Shape with {num_corners} Corners.")

    # 5. Calculate Perspective from a Virtual Bounding Box
    # Even if the room is a hexagon, it sits on a rectangular floor.
    # We use 'minAreaRect' to find the best-fit rectangle for the PERSPECTIVE transform.
    rect = cv2.minAreaRect(arena_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Order the bounding box corners for Homography
    src_rect_pts = order_rect_points(box.reshape(4, 2).astype("float32"))

    # 6. Calculate Dimensions & Homography Matrix M
    (tl, tr, br, bl) = src_rect_pts
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    dst_rect_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # M maps the "Virtual Box" to a flat 2D rectangle
    M = cv2.getPerspectiveTransform(src_rect_pts, dst_rect_pts)

    # 7. Transform the ACTUAL N-Corner Polygon
    # We apply M to the specific shape corners we found earlier
    # Reshape to (N, 1, 2) for perspectiveTransform
    pts_original = polygon_corners.reshape(-1, 1, 2).astype("float32")
    pts_flat = cv2.perspectiveTransform(pts_original, M)
    
    return rgb, mask, polygon_corners, pts_flat.reshape(-1, 2), (width, height)

def plot_flexible_results(rgb, mask, corners_orig, corners_flat, dims):
    width, height = dims
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Mask
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title("1. Red Tape Mask")
    
    # Plot 2: Original Detection (N-Points)
    axes[1].imshow(rgb)
    # Draw the complex shape
    pts = corners_orig.reshape(-1, 2)
    pts = np.append(pts, [pts[0]], axis=0) # Close loop
    axes[1].plot(pts[:, 0], pts[:, 1], 'lime', linewidth=3, label=f'{len(pts)-1} Corners')
    axes[1].scatter(pts[:, 0], pts[:, 1], c='yellow', s=80, zorder=5)
    axes[1].set_title(f"2. Detected {len(pts)-1}-Sided Polygon")
    axes[1].legend()

    # Plot 3: 2D Geometry Map
    # Draw the flattened complex shape
    pts_f = corners_flat
    pts_f = np.append(pts_f, [pts_f[0]], axis=0)
    
    axes[2].plot(pts_f[:, 0], pts_f[:, 1], 'b-', linewidth=3, marker='o')
    axes[2].set_title("3. Final 2D Geometry Map")
    axes[2].set_xlim(-50, width + 50)
    axes[2].set_ylim(-50, height + 50)
    axes[2].invert_yaxis()
    axes[2].set_aspect('equal')
    axes[2].grid(True)

    plt.show()

if __name__ == "__main__":
    try:
        rgb, mask, orig, flat, dims = process_flexible_arena(IMAGE_PATH)
        plot_flexible_results(rgb, mask, orig, flat, dims)
    except Exception as e:
        print(f"Error: {e}")