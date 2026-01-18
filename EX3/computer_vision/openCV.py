import cv2
import numpy as np

# --- 1. SETUP DESTINATION POINTS ---
# We define the coordinates based on your diagram dimensions.
# We'll use a scale factor so the output image isn't too small (1 unit = 5 pixels).
SCALE = 5 

# Coordinates are in (x, y) format:
# Pt A: Top-Left (0, 0)
# Pt B: Top-Right (190, 0)
# Pt C: Bottom-Right (190, 237)
# Pt D: Bottom-Inner (The corner where the straight bottom edge meets the angle) -> (190 - 101, 237) = (89, 237)
# Pt E: Left-Inner (The corner where the straight left edge meets the angle) -> (0, 148)

dst_points = np.array([
    [0, 0],         # Top-Left
    [190, 0],       # Top-Right
    [190, 237],     # Bottom-Right
    [89, 237],      # Bottom-Inner (Start of angle on bottom)
    [0, 148]        # Left-Inner (Start of angle on left)
], dtype='float32') * SCALE

# Calculate output image size based on bounds
max_width = int(190 * SCALE)
max_height = int(237 * SCALE)

# --- 2. INTERACTIVE POINT SELECTION ---
src_points = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < 5:
            src_points.append([x, y])
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_display, str(len(src_points)), (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Select 5 Points", img_display)

# Load your image
image_path = 'arena4.jpeg' 
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not found.")
    exit()

img_display = img.copy()

print("--- INSTRUCTIONS ---")
print("Please click the corners of the arena floor in this EXACT order:")
print("1. Top-Left (Far left corner)")
print("2. Top-Right (Far right corner)")
print("3. Bottom-Right (Closest right corner)")
print("4. Bottom-Inner (Where the bottom straight wall meets the diagonal wall)")
print("5. Left-Inner (Where the left straight wall meets the diagonal wall)")
print("--------------------")

cv2.imshow("Select 5 Points", img_display)
cv2.setMouseCallback("Select 5 Points", select_points)

# Wait until 5 points are selected
while True:
    k = cv2.waitKey(1)
    if len(src_points) == 5:
        print("5 points selected. Press any key to calculate homography...")
        cv2.waitKey(0) # Wait for one more key press to proceed
        break
    if k == 27: # Esc to quit
        exit()

cv2.destroyAllWindows()

# --- 3. COMPUTE HOMOGRAPHY & WARP ---
if len(src_points) == 5:
    src_pts_arr = np.array(src_points, dtype='float32')

    # Find Homography Matrix (Using all 5 points for best fit)
    # RANSAC is robust to slight clicking errors
    H, status = cv2.findHomography(src_pts_arr, dst_points, cv2.RANSAC, 5.0)

    # Warp the image
    warped_img = cv2.warpPerspective(img, H, (max_width, max_height))

    # Show results
    cv2.imshow("Original", img)
    cv2.imshow("Top-Down View", warped_img)
    
    # Save the output
    cv2.imwrite("arena_top_down.jpg", warped_img)
    
    print("Homography complete. Image saved as 'arena_top_down.jpg'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()