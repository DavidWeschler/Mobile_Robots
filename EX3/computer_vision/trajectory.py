import cv2
import numpy as np
import matplotlib.pyplot as plt  # <--- IMPORT FOR PLOTTING

# --- 1. SETTINGS ---
video_source = 'robot_trajectory.mp4'
min_area_size = 500
display_scale = 0.5
warmup_frames = 60

# --- 2. INITIALIZATION ---
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

back_sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

trajectory_points = []
arena_contours = None
frame_count = 0

# --- 3. MAIN LOOP ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)

        # Warmup Logic
        frame_count += 1
        if frame_count < warmup_frames:
            back_sub.apply(frame)
            cv2.imshow("Robot Mapping & Tracking", frame)
            cv2.waitKey(30)
            continue

        # A. Automatic Arena Mapping
        if arena_contours is None:
            gray_static = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_static = cv2.GaussianBlur(gray_static, (5, 5), 0)
            edges_static = cv2.Canny(blurred_static, 30, 100)
            contours_static, _ = cv2.findContours(edges_static, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_static:
                arena_contours = max(contours_static, key=cv2.contourArea)

        # B. Robot Tracking
        fg_mask = back_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, (5, 5))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            robot_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(robot_contour) > min_area_size:
                M = cv2.moments(robot_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    trajectory_points.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # C. Visualization on Video
        # if arena_contours is not None:
        #     cv2.drawContours(frame, [arena_contours], -1, (255, 0, 0), 2)

        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)

        cv2.imshow("Robot Mapping & Tracking", frame)
        if cv2.waitKey(30) == 27: # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# --- 4. POST-RUN PLOTTING ---
print(f"Tracking finished. Total points collected: {len(trajectory_points)}")

if len(trajectory_points) > 0:
    # Convert list to numpy array for easier plotting
    traj_arr = np.array(trajectory_points)
    
    plt.figure(figsize=(10, 8))
    
    # 1. Plot the Trajectory
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], color='red', linewidth=2, label='Robot Path')
    plt.scatter(traj_arr[:, 0], traj_arr[:, 1], color='darkred', s=10) # Dots for each frame
    
    # # 2. Plot the Arena Walls (if found)
    # if arena_contours is not None:
    #     # Reshape contour from (N, 1, 2) to (N, 2)
    #     wall_pts = arena_contours.reshape(-1, 2)
    #     # Close the loop (connect last point to first)
    #     wall_pts = np.vstack([wall_pts, wall_pts[0]])
        
    #     plt.plot(wall_pts[:, 0], wall_pts[:, 1], color='blue', linewidth=3, label='Arena Walls')
    
    # 3. Formatting the Graph
    plt.title("Final Robot Trajectory")
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    plt.legend()
    plt.grid(True)
    
    # IMPORTANT: Images have (0,0) at Top-Left, but Plots have (0,0) at Bottom-Left.
    # We must invert the Y-axis to make it look like the video.
    plt.gca().invert_yaxis()
    plt.axis('equal') # Keep the aspect ratio square
    
    plt.show()
else:
    print("No trajectory points were recorded.")