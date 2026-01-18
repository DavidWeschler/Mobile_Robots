import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CAMERA_INDEX = 2             # Try 0, 1, or 2
OUTPUT_TXT = 'trajectory_real_world.txt'
WARMUP_FRAMES = 60
MIN_AREA_SIZE = 500
DISPLAY_SCALE = 0.5

# --- 1. HELPER FUNCTIONS ---

def capture_good_frame(cap):
    """
    Shows live video until user presses SPACE.
    Ensures the image is bright and focused before we click.
    """
    print("--- STEP 1: CAMERA SETUP ---")
    print("Adjust your camera.")
    print("Press SPACE to freeze the image and start calibration.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading camera.")
            continue
            
        # Resize for display
        display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        
        # Add instructions text
        cv2.putText(display_frame, "Adjust Camera & Light", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press SPACE to Capture", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("CAMERA SETUP", display_frame)
        
        key = cv2.waitKey(1)
        if key == 32: # SPACE bar
            break
        if key == 27: # ESC
            exit()
            
    cv2.destroyWindow("CAMERA SETUP")
    return frame

def get_homography_manual(img):
    """
    Opens the frozen image to select 5 points.
    """
    dst_points = np.array([
        [0, 0],         
        [190, 0],       
        [190, 237],     
        [89, 237],      
        [0, 148]        
    ], dtype='float32')

    src_points = []
    display_img = img.copy()

    # Scale handling for display
    calib_scale = 1.0
    h, w = img.shape[:2]
    # If image is wider than screen (e.g., 1920), scale it down for clicking
    if w > 1200:
        calib_scale = 1200 / w
        display_img = cv2.resize(display_img, (0, 0), fx=calib_scale, fy=calib_scale)

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(src_points) < 5:
                # Map click back to original resolution
                real_x = int(x / calib_scale)
                real_y = int(y / calib_scale)
                src_points.append([real_x, real_y])
                
                # Draw visual feedback
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(display_img, str(len(src_points)), (x+10, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("CALIBRATION: Click 5 Points", display_img)

    cv2.imshow("CALIBRATION: Click 5 Points", display_img)
    cv2.setMouseCallback("CALIBRATION: Click 5 Points", select_points)

    print("\n--- STEP 2: CALIBRATION ---")
    print("Click the 5 corners in this order:")
    print("1. Top-Left")
    print("2. Top-Right")
    print("3. Bottom-Right")
    print("4. Bottom-Inner")
    print("5. Left-Inner")
    
    while True:
        if len(src_points) == 5:
            print("5 points selected. Press ANY KEY to continue...")
            cv2.waitKey(0)
            break
        if cv2.waitKey(1) == 27: # ESC
            exit()
    
    cv2.destroyWindow("CALIBRATION: Click 5 Points")
    H, _ = cv2.findHomography(np.array(src_points, dtype='float32'), dst_points, cv2.RANSAC, 5.0)
    return H

def transform_point(point, H):
    pt_array = np.array([[point]], dtype='float32')
    transformed = cv2.perspectiveTransform(pt_array, H)
    return transformed[0][0]

# --- 2. SETUP ---

print(f"Opening Camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) # Faster loading on Windows

if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open Camera.")
        exit()

# --- PHASE 1: CAPTURE IMAGE ---
# This runs the live feed loop so you can see what you are doing
calibration_frame = capture_good_frame(cap)

# --- PHASE 2: CLICK POINTS ---
# This uses the frozen image you just captured
H = get_homography_manual(calibration_frame)
print("Calibration Complete.")

# --- PHASE 3: TRACKING ---
back_sub = cv2.createBackgroundSubtractorKNN(history=1500, dist2Threshold=400, detectShadows=True)
trajectory_real = []
trajectory_pixels = []
frame_count = 0

with open(OUTPUT_TXT, 'w') as f:
    f.write("Frame, X_cm, Y_cm\n")
    print(f"\n--- STEP 3: TRACKING STARTED ---")
    print(f"Saving to {OUTPUT_TXT}")
    print("Press ESC to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            frame_count += 1

            # WARMUP
            if frame_count < WARMUP_FRAMES:
                back_sub.apply(frame)
                cv2.putText(display_frame, f"Learning Background: {frame_count}/{WARMUP_FRAMES}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow("Robot Tracking", display_frame)
                cv2.waitKey(1)
                continue

            # TRACKING
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

                        real_pt = transform_point((cx, cy), H)
                        rx, ry = real_pt[0], real_pt[1]

                        trajectory_real.append((rx, ry))
                        trajectory_pixels.append((cx, cy))
                        f.write(f"{frame_count}, {rx:.2f}, {ry:.2f}\n")

                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Draw Path
            if len(trajectory_pixels) > 1:
                pts_draw = trajectory_pixels[-50:] # Draw last 50 points
                for i in range(1, len(pts_draw)):
                    cv2.line(frame, pts_draw[i-1], pts_draw[i], (0, 0, 255), 3)

            # Update Display
            display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("Robot Tracking", display_frame)

            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# --- PLOTTING ---
if len(trajectory_real) > 0:
    real_arr = np.array(trajectory_real)
    plt.figure(figsize=(10, 10))
    plt.plot(real_arr[:, 0], real_arr[:, 1], 'r-', linewidth=2, label='Path')
    
    walls = np.array([[0, 0], [190, 0], [190, 237], [89, 237], [0, 148], [0, 0]])
    plt.plot(walls[:, 0], walls[:, 1], 'b-', linewidth=3, label='Walls')
    
    plt.legend()
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()