import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Image
image_path = 'arena.jpeg'
# 
image_path = 'arena_top_down.jpg'

img = cv2.imread(image_path)
if img is None:
    print("Error: Image not found.")
    exit()

# 2. Preprocessing
# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaussianBlur = 5
# Apply Gaussian Blur to reduce noise (texture of concrete)
# (5, 5) is the kernel size. Larger numbers = more blur.
blurred = cv2.GaussianBlur(gray, (gaussianBlur, gaussianBlur), 0)

# 3. Edge Detection (Canny)
# 50 and 150 are thresholds. 
# Pixels with gradient > 150 are edges. < 50 are not. Between is conditional.
edges = cv2.Canny(blurred, 50, 150)

# 4. Find Contours
# RETR_EXTERNAL gets only the outer boundaries (ignores tape marks inside)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No contours found. Try adjusting Canny thresholds.")
    exit()

# 5. Find the Largest Contour (The Arena Floor)
# We assume the arena floor is the biggest shape in the picture
largest_contour = max(contours, key=cv2.contourArea)

# 6. Simplify the Contour (Polygonal Approximation)
# Epsilon is the accuracy parameter. 
# 0.01 * perimeter means "Ignore wiggles smaller than 1% of the total length"
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

print(f"Detected {len(approx_polygon)} wall corners.")

# 7. Extract coordinates for Plotting
# approx_polygon is shape (N, 1, 2), we need (N, 2)
points = approx_polygon.reshape(-1, 2)

# Close the loop for the plot (connect last point to first)
points = np.vstack([points, points[0]]) 

# 8. Plot using Matplotlib
plt.figure(figsize=(10, 8))
plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Detected Walls')
plt.scatter(points[:, 0], points[:, 1], color='red', zorder=5) # Show corners

# Invert Y axis because images use (0,0) at top-left, but plots use bottom-left
plt.gca().invert_yaxis() 
plt.title("Arena Wall Map (Auto-Detected)")
plt.xlabel("Pixels X")
plt.ylabel("Pixels Y")
plt.grid(True)
plt.legend()
plt.show()

# Optional: Show the Canny result to debug
cv2.namedWindow("Debug: Edges", cv2.WINDOW_NORMAL)
cv2.imshow("Debug: Edges", edges)
cv2.resizeWindow("Debug: Edges", 600, 400)
cv2.waitKey(0)
cv2.destroyAllWindows()