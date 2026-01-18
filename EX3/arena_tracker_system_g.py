import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
CAMERA_INDEX = 0             # 0 is usually the default webcam. Try 1 or 2 if using DroidCam/External.
OUTPUT_TXT = 'trajectory_real_world.txt'
WARMUP_FRAMES = 60
MIN_AREA_SIZE = 500
DISPLAY_SCALE = 0.5          # Scale down the view for speed/screen fit

# --- 1. HELPER FUNCTIONS ---

def get_homography_manual(img):
    """
    Opens an interactive window to select 5 points and calculates H.
    """
    # Points: Top-Left, Top-Right, Bottom-Right, Bottom-Inner, Left-Inner
    # Dimensions based on your arena specs (in cm)
    dst_points = np.array([
        [0, 0],         
        [190, 0],       
        [190, 237],     
        [89, 237],      
        [0, 148]        
    ], dtype='float32')

    src_points = []
    # Make a copy so we don't draw over the original image used for tracking later (if needed)
    display_img = img.copy()

    # If the camera image is huge, resize specifically for the calibration window
    # but we must track the scale to map clicks back to the original image size
    calib_scale = 1.0
    h, w = img.shape[:2]
    if w > 1280:
        calib_scale = 1280 / w
        display_img = cv2.resize(display_img, (0, 0), fx=calib_scale, fy=calib_scale)

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(src_points) < 5:
                # Store the point (x, y)
                # If we scaled the image, we must scale the click UP to match original resolution
                real_x = int(x / calib_scale)
                real_y = int(y / calib_scale)
                src_points.append([real_x, real_y])
                
                # Draw on the display image
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(display_img, str(len(src_points)), (x+10, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("CALIBRATION: Click 5 Points", display_img)

    cv2.imshow("CALIBRATION: Click 5 Points", display_img)
    cv2.setMouseCallback("CALIBRATION: Click 5 Points", select_points)

    print("--- CALIBRATION ---")
    print("Click the 5 corners in this order:")
    print("1. Top-Left")
    print("2. Top-Right")
    print("3. Bottom-Right")
    print("4. Bottom-Inner (Angle start)")
    print("5. Left-Inner (Angle start)")
    
    while True:
        if len(src_points) == 5:
            print("5 points selected. Press ANY KEY to continue...")
            cv2.waitKey(0)
            break
        key = cv2.waitKey(1)
        if key == 27: # ESC
            exit()
    
    cv2.destroyWindow("CALIBRATION: Click 5 Points")
    
    # Calculate Homography using the original resolution points
    H, _ = cv2.findHomography(np.array(src_points, dtype='float32'), dst_points, cv2.RANSAC, 5.0)
    return H

def transform_point(point, H):
    """
    Converts a single (x, y) pixel point to real-world coordinates using H.
    """
    pt_array = np.array([[point]], dtype='float32')
    transformed = cv2.perspectiveTransform(pt_array, H)
    return transformed[0][0] # Returns (x, y)

# --- 2. SETUP & CALIBRATION ---

print(f"Opening Camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) # CAP_DSHOW often loads faster on Windows

if not cap.isOpened():
    # Fallback if CAP_DSHOW fails
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open Camera {CAMERA_INDEX}")
        exit()

# Allow camera to warm up brightness/focus
print("Warming up camera sensor...")
for _ in range(30):
    cap.read()

# Grab a SINGLE clear frame for calibration
print("Capturing calibration frame...")
ret, calibration_frame = cap.read()
if not ret:
    print("Error: Could not read from camera.")
    exit()

# RUN CALIBRATION (User clicks points on the still image)
H = get_homography_manual(calibration_frame)
print("Homography Matrix calculated.")

# --- 3. LIVE TRACKING LOOP ---

back_sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
trajectory_real = [] # Stores real (x, y)
trajectory_pixels = [] # Stores pixel (u, v) for visual
frame_count = 0

# Open file for writing
with open(OUTPUT_TXT, 'w') as f:
    f.write("Frame, X_cm, Y_cm\n") # Header

    print(f"Starting LIVE tracking... Saving to {OUTPUT_TXT}")
    print("Press ESC to stop and plot results.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera disconnected or frame error.")
                break
            
            # Create a display version (resized)
            display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            
            frame_count += 1

            # WARMUP: Skip tracking for first N frames to let background settle
            # This is crucial for live cameras as auto-exposure adjusts
            if frame_count < WARMUP_FRAMES:
                back_sub.apply(frame) # Teach background
                cv2.putText(display_frame, f"Warmup: {frame_count}/{WARMUP_FRAMES}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("Robot Tracking", display_frame)
                cv2.waitKey(1)
                continue

            # ROBOT DETECTION
            fg_mask = back_sub.apply(frame)
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, (5,5))

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                robot_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(robot_contour) > MIN_AREA_SIZE:
                    M = cv2.moments(robot_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 1. Transform to Real World
                        real_pt = transform_point((cx, cy), H)
                        rx, ry = real_pt[0], real_pt[1]

                        # 2. Save Data
                        trajectory_real.append((rx, ry))
                        trajectory_pixels.append((cx, cy))
                        f.write(f"{frame_count}, {rx:.2f}, {ry:.2f}\n")

                        # 3. Visuals (Draw on original frame)
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # DRAW PATH ON VIDEO
            if len(trajectory_pixels) > 1:
                # Only draw the last 50 points to keep video smooth, or all if you prefer
                points_to_draw = trajectory_pixels[-50:] 
                for i in range(1, len(points_to_draw)):
                    cv2.line(frame, points_to_draw[i-1], points_to_draw[i], (0, 0, 255), 3)

            # Update Display
            display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("Robot Tracking", display_frame)

            if cv2.waitKey(1) == 27: # ESC
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# --- 4. FINAL PLOTTING ---
print("Tracking complete. Generating plot...")

if len(trajectory_real) > 0:
    real_arr = np.array(trajectory_real)
    
    plt.figure(figsize=(10, 10))
    
    # Plot Trajectory
    plt.plot(real_arr[:, 0], real_arr[:, 1], 'r-', linewidth=2, label='Robot Path')
    plt.scatter(real_arr[:, 0], real_arr[:, 1], c='darkred', s=5)

    # Plot Arena Boundary (Using the Manual Dimensions)
    walls = np.array([
        [0, 0], [190, 0], [190, 237], [89, 237], [0, 148], [0, 0]
    ])
    plt.plot(walls[:, 0], walls[:, 1], 'b-', linewidth=3, label='Arena Walls (cm)')

    plt.title("Real-World Trajectory (Centimeters)")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.gca().invert_yaxis() # Match image coordinates logic
    
    plt.savefig("final_trajectory_plot.png")
    plt.show()
    print("Plot saved as final_trajectory_plot.png")
else:
    print("No trajectory points found.")