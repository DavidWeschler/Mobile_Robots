"""
Arena Tracker System - Complete Robot Tracking & Communication
==============================================================
This script combines:
1. Camera capture (from computer/phone camera)
2. Arena detection (red tape walls and corners)
3. Robot trajectory tracking (real-time)
4. Bluetooth communication to NXT robot

The system:
- Detects the arena boundaries using red tape
- Tracks the robot's position in real-time
- Sends arena data and trajectory to the robot periodically
- Displays everything on screen with visualization

Author: Combined from multiple EX3 scripts
"""

import cv2
import numpy as np
import socket
import struct
import time
import threading
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """All configuration settings in one place"""
    # Camera Settings
    CAMERA_INDEX: int = 2  # 0, 1, or 2 (phone link via DroidCam is usually 2)
    CAMERA_BACKEND: int = cv2.CAP_MSMF  # CAP_DSHOW or CAP_MSMF
    DISPLAY_SCALE: float = 0.7  # Scale factor for display
    
    # Arena Calibration Settings (Manual Corner Selection)
    # These are the REAL WORLD dimensions of your arena in cm
    # The user-clicked corners will be mapped to these coordinates
    ARENA_SCALE: int = 5  # 1 cm = 5 pixels in the 2D map
    # Destination points for homography (arena corners in real-world cm)
    # Default: 5-corner arena shape (modify to match your arena)
    # Pt 1: Top-Left (0, 0)
    # Pt 2: Top-Right (190, 0)
    # Pt 3: Bottom-Right (190, 237)
    # Pt 4: Bottom-Inner (89, 237) - where bottom meets diagonal
    # Pt 5: Left-Inner (0, 148) - where left meets diagonal
    ARENA_DST_POINTS: tuple = (
        (0, 0),       # Top-Left
        (190, 0),     # Top-Right  
        (190, 237),   # Bottom-Right
        (89, 237),    # Bottom-Inner
        (0, 148)      # Left-Inner
    )
    NUM_CALIBRATION_POINTS: int = 5  # Number of corners to click
    
    # Arena Detection Settings (for automatic detection - backup)
    MIN_ARENA_AREA: int = 5000  # Minimum contour area to be considered arena
    POLYGON_EPSILON: float = 0.015  # Polygon approximation accuracy (1.5%)
    
    # Robot Tracking Settings
    BG_HISTORY: int = 5000  # Background subtractor history
    BG_THRESHOLD: float = 400  # Background subtractor threshold
    MIN_ROBOT_AREA: int = 300  # Minimum contour area to be robot
    WARMUP_FRAMES: int = 60  # Frames to wait before tracking
    MAX_TRAJECTORY_POINTS: int = 500  # Maximum trajectory points to keep
    
    # Bluetooth Settings
    NXT_ADDRESS: str = "00:16:53:0A:A0:2D"  # NXT Bluetooth address
    NXT_PORT: int = 1  # RFCOMM port
    MAILBOX_ARENA: int = 1  # Mailbox for arena data
    MAILBOX_PATH: int = 2  # Mailbox for path data
    MAILBOX_STATUS: int = 3  # Mailbox for status messages
    
    # Communication Settings
    SEND_INTERVAL: float = 2.0  # Seconds between data sends to robot
    MAX_CORNERS_TO_SEND: int = 8  # Max corners to send (NXT screen limit)
    MAX_PATH_POINTS_TO_SEND: int = 20  # Max path points to send

# =============================================================================
# ARENA DETECTOR CLASS (with Manual Corner Selection)
# =============================================================================

class ArenaDetector:
    """Detects arena boundaries using manual corner selection and homography"""
    
    def __init__(self, config: Config):
        self.config = config
        self.corners_original = None  # Corners in camera image space
        self.corners_flat = None      # Corners in 2D map space
        self.homography_matrix = None
        self.arena_dimensions = None
        self.walls = []
        self.is_calibrated = False
        
        # Setup destination points from config
        self._setup_destination_points()
    
    def _setup_destination_points(self):
        """Setup the destination points for homography from config"""
        scale = self.config.ARENA_SCALE
        self.dst_points = np.array(self.config.ARENA_DST_POINTS, dtype='float32') * scale
        
        # Calculate arena dimensions from destination points
        max_x = max(pt[0] for pt in self.config.ARENA_DST_POINTS)
        max_y = max(pt[1] for pt in self.config.ARENA_DST_POINTS)
        self.arena_dimensions = (int(max_x * scale), int(max_y * scale))
    
    def calibrate_with_points(self, src_points: List[Tuple[int, int]]) -> bool:
        """
        Calibrates arena using manually selected source points.
        src_points: List of (x, y) tuples clicked by user in camera image.
        These will be mapped to the destination points defined in config.
        """
        try:
            num_points = len(src_points)
            expected_points = self.config.NUM_CALIBRATION_POINTS
            
            if num_points != expected_points:
                print(f"❌ Expected {expected_points} points, got {num_points}")
                return False
            
            # Convert to numpy arrays
            src_pts_arr = np.array(src_points, dtype='float32')
            
            # Store original corners (in camera space)
            self.corners_original = src_pts_arr.reshape(-1, 1, 2).astype(np.int32)
            
            # Compute homography matrix using RANSAC for robustness
            self.homography_matrix, status = cv2.findHomography(
                src_pts_arr, 
                self.dst_points, 
                cv2.RANSAC, 
                5.0
            )
            
            if self.homography_matrix is None:
                print("❌ Failed to compute homography matrix")
                return False
            
            # The flat corners ARE the destination points
            self.corners_flat = self.dst_points.copy()
            
            # Extract walls as line segments
            self.walls = []
            for i in range(len(self.corners_flat)):
                p1 = self.corners_flat[i]
                p2 = self.corners_flat[(i + 1) % len(self.corners_flat)]
                self.walls.append((p1.tolist(), p2.tolist()))
            
            self.is_calibrated = True
            print(f"✅ Arena calibrated with {num_points} corners!")
            print(f"   Arena dimensions: {self.arena_dimensions[0]}x{self.arena_dimensions[1]} pixels")
            return True
            
        except Exception as e:
            print(f"❌ Calibration error: {e}")
            return False
    
    def get_red_mask(self, img_hsv: np.ndarray) -> np.ndarray:
        """Isolates red tape from HSV image (for automatic detection backup)"""
        lower1, upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower2, upper2 = np.array([160, 100, 100]), np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower1, upper1)
        mask2 = cv2.inRange(img_hsv, lower2, upper2)
        return mask1 + mask2
    
    def detect_arena_auto(self, frame: np.ndarray) -> bool:
        """
        Automatic arena detection using red tape (backup method).
        Returns True if successful.
        """
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = self.get_red_mask(hsv)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
            
            arena_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(arena_contour) < self.config.MIN_ARENA_AREA:
                return False
            
            epsilon = self.config.POLYGON_EPSILON * cv2.arcLength(arena_contour, True)
            polygon_corners = cv2.approxPolyDP(arena_contour, epsilon, True)
            
            # Extract corner points and use them for calibration
            corners = [(int(pt[0][0]), int(pt[0][1])) for pt in polygon_corners]
            
            # If we have the right number of corners, calibrate
            if len(corners) == self.config.NUM_CALIBRATION_POINTS:
                return self.calibrate_with_points(corners)
            else:
                print(f"⚠️ Auto-detected {len(corners)} corners, expected {self.config.NUM_CALIBRATION_POINTS}")
                return False
            
        except Exception as e:
            print(f"❌ Auto detection error: {e}")
            return False
    
    def transform_point(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Transforms a point from camera space to arena 2D space"""
        if self.homography_matrix is None:
            return None
        
        pt = np.array([[[point[0], point[1]]]], dtype="float32")
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
    
    def draw_arena(self, frame: np.ndarray) -> np.ndarray:
        """Draws detected arena on frame"""
        if self.corners_original is None:
            return frame
        
        # Draw polygon connecting corners
        pts = self.corners_original.reshape(-1, 2)
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % len(pts)])
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
        
        # Draw corner points with labels
        for i, pt in enumerate(pts):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)
            cv2.putText(frame, str(i+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame

# =============================================================================
# ROBOT TRACKER CLASS
# =============================================================================

class RobotTracker:
    """Tracks robot position using background subtraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.back_sub = cv2.createBackgroundSubtractorKNN(
            history=config.BG_HISTORY,
            dist2Threshold=config.BG_THRESHOLD,
            detectShadows=True
        )
        self.trajectory_points = deque(maxlen=config.MAX_TRAJECTORY_POINTS)
        self.trajectory_points_flat = deque(maxlen=config.MAX_TRAJECTORY_POINTS)
        self.frame_count = 0
        self.current_position = None
        self.current_position_flat = None
        self.is_tracking = False
    
    def update(self, frame: np.ndarray, arena_detector: Optional[ArenaDetector] = None) -> Tuple[int, int]:
        """
        Updates tracker with new frame.
        Returns robot position or None if not detected.
        """
        self.frame_count += 1
        
        # Warmup phase - just update background model
        if self.frame_count < self.config.WARMUP_FRAMES:
            self.back_sub.apply(frame)
            return None
        
        self.is_tracking = True
        
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest moving object (robot)
        robot_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(robot_contour) < self.config.MIN_ROBOT_AREA:
            return None
        
        # Calculate centroid
        M = cv2.moments(robot_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        self.current_position = (cx, cy)
        self.trajectory_points.append((cx, cy))
        
        # Transform to arena 2D space if calibrated
        if arena_detector and arena_detector.is_calibrated:
            flat_pos = arena_detector.transform_point((cx, cy))
            if flat_pos:
                self.current_position_flat = flat_pos
                self.trajectory_points_flat.append(flat_pos)
        
        return (cx, cy)
    
    def draw_trajectory(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Draws trajectory on frame"""
        points = list(self.trajectory_points)
        
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], color, 2)
        
        # Draw current position
        if self.current_position:
            cv2.circle(frame, self.current_position, 12, (0, 255, 0), -1)
            cv2.circle(frame, self.current_position, 14, (255, 255, 255), 2)
        
        return frame
    
    def get_warmup_progress(self) -> float:
        """Returns warmup progress (0.0 to 1.0)"""
        return min(1.0, self.frame_count / self.config.WARMUP_FRAMES)

# =============================================================================
# NXT BLUETOOTH COMMUNICATOR CLASS
# =============================================================================

class NXTCommunicator:
    """Handles Bluetooth communication with NXT robot"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sock = None
        self.is_connected = False
        self.last_send_time = 0
        self.send_lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connects to NXT via Bluetooth"""
        try:
            self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.settimeout(5.0)
            
            print(f"📡 Connecting to NXT at {self.config.NXT_ADDRESS}...")
            self.sock.connect((self.config.NXT_ADDRESS, self.config.NXT_PORT))
            self.is_connected = True
            print("✅ Connected to NXT!")
            return True
            
        except OSError as e:
            print(f"❌ Connection error: {e}")
            print("Make sure your NXT is turned on and Bluetooth is paired.")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnects from NXT"""
        if self.sock:
            self.sock.close()
            self.is_connected = False
            print("📴 Disconnected from NXT")
    
    def send_message(self, mailbox: int, message: str) -> bool:
        """Sends a message to NXT mailbox"""
        if not self.is_connected:
            return False
        
        try:
            with self.send_lock:
                # NXT Direct Command: MessageWrite (0x09)
                msg_bytes = message.encode()[:58] + b'\x00'  # Max 58 chars + null terminator
                cmd = bytes([0x00, 0x09, mailbox, len(msg_bytes)]) + msg_bytes
                packet = struct.pack('<H', len(cmd)) + cmd
                self.sock.send(packet)
                
                # Read response (non-blocking with short timeout)
                self.sock.settimeout(0.1)
                try:
                    response = self.sock.recv(64)
                except socket.timeout:
                    pass
                self.sock.settimeout(5.0)
                
                return True
        except Exception as e:
            print(f"❌ Send error: {e}")
            return False
    
    def send_arena_data(self, corners: List[Tuple[float, float]], 
                        dimensions: Tuple[int, int]) -> bool:
        """Sends arena corners and dimensions to robot"""
        # Format: "W:width,H:height,C:x1,y1;x2,y2;..."
        w, h = dimensions
        
        # Limit corners to send
        corners_to_send = corners[:self.config.MAX_CORNERS_TO_SEND]
        corners_str = ";".join([f"{int(x)},{int(y)}" for x, y in corners_to_send])
        
        msg = f"A:{w},{h}|{len(corners_to_send)}|{corners_str}"
        return self.send_message(self.config.MAILBOX_ARENA, msg)
    
    def send_path_data(self, path_points: List[Tuple[int, int]], 
                       current_pos: Optional[Tuple[int, int]]) -> bool:
        """Sends robot path data to robot"""
        # Format: "P:cx,cy|n|x1,y1;x2,y2;..."
        
        # Subsample path points to fit
        path_to_send = list(path_points)
        if len(path_to_send) > self.config.MAX_PATH_POINTS_TO_SEND:
            step = len(path_to_send) // self.config.MAX_PATH_POINTS_TO_SEND
            path_to_send = path_to_send[::step][:self.config.MAX_PATH_POINTS_TO_SEND]
        
        # Current position
        pos_str = f"{current_pos[0]},{current_pos[1]}" if current_pos else "0,0"
        
        # Path points
        path_str = ";".join([f"{int(x)},{int(y)}" for x, y in path_to_send])
        
        msg = f"P:{pos_str}|{len(path_to_send)}|{path_str}"
        return self.send_message(self.config.MAILBOX_PATH, msg)
    
    def send_status(self, status: str) -> bool:
        """Sends status message to robot"""
        return self.send_message(self.config.MAILBOX_STATUS, status)
    
    def should_send(self) -> bool:
        """Checks if enough time has passed for next send"""
        current_time = time.time()
        if current_time - self.last_send_time >= self.config.SEND_INTERVAL:
            self.last_send_time = current_time
            return True
        return False

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class ArenaTrackerSystem:
    """Main application combining all components"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.arena_detector = ArenaDetector(self.config)
        self.robot_tracker = RobotTracker(self.config)
        self.nxt_comm = NXTCommunicator(self.config)
        
        self.cap = None
        self.running = False
        self.calibration_mode = True
        
        # Manual calibration state
        self.calibration_points = []
        self.calibration_frame = None
        self.calibration_display = None
        
    def initialize_camera(self) -> bool:
        """Initializes the camera"""
        print(f"📷 Opening camera {self.config.CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX, self.config.CAMERA_BACKEND)
        
        if not self.cap.isOpened():
            print("❌ Could not open camera!")
            return False
        
        print("✅ Camera opened successfully!")
        return True
    
    def initialize_bluetooth(self) -> bool:
        """Initializes Bluetooth connection to NXT"""
        return self.nxt_comm.connect()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for manual corner selection (works with live feed)"""
        if event == cv2.EVENT_LBUTTONDOWN:
            num_points = self.config.NUM_CALIBRATION_POINTS
            if len(self.calibration_points) < num_points:
                self.calibration_points.append((x, y))
                print(f"   Point {len(self.calibration_points)}: ({x}, {y})")
    
    def run_manual_calibration(self) -> bool:
        """
        Runs the manual calibration process where user clicks arena corners.
        Shows LIVE camera feed so user can see the arena.
        Returns True if calibration successful.
        """
        print("\n" + "="*60)
        print("         MANUAL ARENA CALIBRATION")
        print("="*60)
        print(f"\nPlease click the {self.config.NUM_CALIBRATION_POINTS} corners of the arena in order:")
        print("")
        for i, pt in enumerate(self.config.ARENA_DST_POINTS):
            print(f"  {i+1}. Point at real-world position ({pt[0]}, {pt[1]}) cm")
        print("\n" + "-"*60)
        print("Instructions:")
        print("  - Click corners in the EXACT order listed above")
        print("  - Press 'r' to reset and start over")
        print("  - Press 'ESC' to cancel calibration")
        print("  - After selecting all points, press ENTER to confirm")
        print("-"*60 + "\n")
        
        # Warm up the camera - read a few frames to let it adjust
        print("📷 Warming up camera...")
        for _ in range(30):
            self.cap.read()
        print("✅ Camera ready!")
        
        self.calibration_points = []
        
        # Setup mouse callback
        window_name = "Arena Calibration - Click Corners (LIVE)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        num_points = self.config.NUM_CALIBRATION_POINTS
        
        while True:
            # Capture LIVE frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                continue
            
            # Resize frame
            frame = cv2.resize(frame, (0, 0), 
                              fx=self.config.DISPLAY_SCALE, 
                              fy=self.config.DISPLAY_SCALE)
            
            display = frame.copy()
            
            # Draw instructions overlay
            cv2.rectangle(display, (5, 5), (450, 90), (0, 0, 0), -1)
            cv2.putText(display, f"Click {num_points} corners in order", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Press 'r' to reset, ESC to cancel", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "Press ENTER when done selecting all points", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw already selected points on live feed
            for i, pt in enumerate(self.calibration_points):
                cv2.circle(display, pt, 8, (0, 0, 255), -1)
                cv2.circle(display, pt, 10, (255, 255, 255), 2)
                cv2.putText(display, str(i+1), (pt[0]+12, pt[1]-12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw lines connecting selected points
            if len(self.calibration_points) > 1:
                for i in range(len(self.calibration_points) - 1):
                    cv2.line(display, self.calibration_points[i], self.calibration_points[i+1], (0, 255, 0), 2)
            
            # Show point count
            h = display.shape[0]
            status_color = (0, 255, 0) if len(self.calibration_points) == num_points else (0, 255, 255)
            cv2.putText(display, f"Points: {len(self.calibration_points)}/{num_points}", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show confirmation message when all points selected
            if len(self.calibration_points) == num_points:
                cv2.putText(display, "All points selected! Press ENTER to confirm...", 
                           (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            # ESC to cancel
            if key == 27:
                print("❌ Calibration cancelled by user")
                cv2.destroyWindow(window_name)
                return False
            
            # 'r' to reset
            if key == ord('r'):
                print("🔄 Resetting calibration points...")
                self.calibration_points = []
            
            # ENTER to confirm (when all points selected)
            if key == 13 and len(self.calibration_points) == num_points:  # 13 is Enter key
                break
        
        cv2.destroyWindow(window_name)
        
        # Perform calibration with selected points
        success = self.arena_detector.calibrate_with_points(self.calibration_points)
        
        if success:
            print("\n✅ Calibration successful!")
            self.nxt_comm.send_status("Arena OK")
        else:
            print("\n❌ Calibration failed!")
        
        return success
    
    def draw_side_panel(self, height: int) -> np.ndarray:
        """
        Creates a side panel with UI info and 2D map.
        This keeps the camera view unobstructed.
        """
        panel_width = 320
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        
        # Title with background
        cv2.rectangle(panel, (5, 5), (panel_width - 5, 35), (50, 50, 50), -1)
        cv2.putText(panel, "Arena Tracker System", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Status section
        y_offset = 55
        
        # Arena status
        arena_status = "CALIBRATED" if self.arena_detector.is_calibrated else "NOT CALIBRATED"
        arena_color = (0, 255, 0) if self.arena_detector.is_calibrated else (0, 0, 255)
        cv2.putText(panel, f"Arena: {arena_status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, arena_color, 1)
        y_offset += 25
        
        # Tracking status
        if not self.robot_tracker.is_tracking:
            progress = self.robot_tracker.get_warmup_progress()
            cv2.putText(panel, f"Warmup: {int(progress * 100)}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            track_status = "TRACKING" if self.robot_tracker.current_position else "SEARCHING"
            track_color = (0, 255, 0) if self.robot_tracker.current_position else (255, 165, 0)
            cv2.putText(panel, f"Robot: {track_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)
        y_offset += 25
        
        # Bluetooth status
        bt_status = "CONNECTED" if self.nxt_comm.is_connected else "DISCONNECTED"
        bt_color = (0, 255, 0) if self.nxt_comm.is_connected else (0, 0, 255)
        cv2.putText(panel, f"Bluetooth: {bt_status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bt_color, 1)
        y_offset += 25
        
        # Path points count
        path_count = len(self.robot_tracker.trajectory_points)
        cv2.putText(panel, f"Path points: {path_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Current position (if available)
        if self.robot_tracker.current_position_flat:
            pos = self.robot_tracker.current_position_flat
            cv2.putText(panel, f"Position: ({pos[0]}, {pos[1]})", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 35
        
        # Separator line
        cv2.line(panel, (10, y_offset), (panel_width - 10, y_offset), (100, 100, 100), 1)
        y_offset += 10
        
        # 2D Map section
        cv2.putText(panel, "2D Map View", (10, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        # Draw 2D map in the panel
        map_size = min(panel_width - 20, height - y_offset - 80)
        if map_size > 50:
            map_view = self.draw_2d_map((map_size, map_size))
            panel[y_offset:y_offset + map_size, 10:10 + map_size] = map_view
            y_offset += map_size + 10
        
        # Instructions at bottom
        cv2.putText(panel, "Controls:", (10, height - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(panel, "'c' - Recalibrate arena", (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(panel, "'r' - Reset tracking", (10, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(panel, "'b' - Reconnect BT  ESC - Quit & Plot", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return panel
    
    def draw_2d_map(self, size: Tuple[int, int] = (300, 300)) -> np.ndarray:
        """Creates a 2D map visualization"""
        map_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        if not self.arena_detector.is_calibrated:
            cv2.putText(map_img, "Arena not calibrated", (10, size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return map_img
        
        # Scale factor to fit arena in map view
        arena_w, arena_h = self.arena_detector.arena_dimensions
        scale = min((size[0] - 40) / arena_w, (size[1] - 40) / arena_h)
        offset_x, offset_y = 20, 20
        
        # Draw walls
        corners = self.arena_detector.corners_flat
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            pt1 = (int(p1[0] * scale + offset_x), int(p1[1] * scale + offset_y))
            pt2 = (int(p2[0] * scale + offset_x), int(p2[1] * scale + offset_y))
            cv2.line(map_img, pt1, pt2, (0, 0, 255), 2)  # Red walls
        
        # Draw corners
        for i, pt in enumerate(corners):
            x = int(pt[0] * scale + offset_x)
            y = int(pt[1] * scale + offset_y)
            cv2.circle(map_img, (x, y), 5, (255, 0, 255), -1)
        
        # Draw trajectory
        path_points = list(self.robot_tracker.trajectory_points_flat)
        if len(path_points) > 1:
            for i in range(1, len(path_points)):
                p1 = path_points[i-1]
                p2 = path_points[i]
                pt1 = (int(p1[0] * scale + offset_x), int(p1[1] * scale + offset_y))
                pt2 = (int(p2[0] * scale + offset_x), int(p2[1] * scale + offset_y))
                cv2.line(map_img, pt1, pt2, (255, 255, 0), 1)  # Yellow path
        
        # Draw current robot position
        if self.robot_tracker.current_position_flat:
            pos = self.robot_tracker.current_position_flat
            x = int(pos[0] * scale + offset_x)
            y = int(pos[1] * scale + offset_y)
            cv2.circle(map_img, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(map_img, (x, y), 10, (255, 255, 255), 2)
        
        return map_img
    
    def show_final_plots(self):
        """
        Shows matplotlib plots of the 2D map and robot trajectory.
        Called when user presses ESC to quit.
        """
        print("\n📊 Generating final plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Plot 1: 2D Arena Map ---
        ax1 = axes[0]
        ax1.set_title("Arena 2D Map", fontsize=14, fontweight='bold')
        
        if self.arena_detector.is_calibrated:
            corners = self.arena_detector.corners_flat
            
            # Draw walls (connect corners)
            for i in range(len(corners)):
                p1 = corners[i]
                p2 = corners[(i + 1) % len(corners)]
                ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3, label='Walls' if i == 0 else '')
            
            # Draw corners
            corner_x = [c[0] for c in corners]
            corner_y = [c[1] for c in corners]
            ax1.scatter(corner_x, corner_y, c='purple', s=100, zorder=5, label='Corners')
            
            # Label corners
            for i, (cx, cy) in enumerate(zip(corner_x, corner_y)):
                ax1.annotate(f'C{i+1}', (cx, cy), textcoords="offset points", 
                            xytext=(5, 5), fontsize=10, color='purple')
            
            # Set axis properties
            arena_w, arena_h = self.arena_detector.arena_dimensions
            ax1.set_xlim(-50, arena_w + 50)
            ax1.set_ylim(-50, arena_h + 50)
        else:
            ax1.text(0.5, 0.5, 'Arena not calibrated', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12, color='gray')
        
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.invert_yaxis()  # Match image coordinates
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # --- Plot 2: Robot Trajectory ---
        ax2 = axes[1]
        ax2.set_title("Robot Trajectory", fontsize=14, fontweight='bold')
        
        path_points = list(self.robot_tracker.trajectory_points_flat)
        
        if self.arena_detector.is_calibrated:
            # Draw arena walls (lighter, as background)
            corners = self.arena_detector.corners_flat
            for i in range(len(corners)):
                p1 = corners[i]
                p2 = corners[(i + 1) % len(corners)]
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.5)
        
        if len(path_points) > 0:
            traj_arr = np.array(path_points)
            
            # Plot trajectory line
            ax2.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, label='Path')
            
            # Plot trajectory points
            ax2.scatter(traj_arr[:, 0], traj_arr[:, 1], c=range(len(traj_arr)), 
                       cmap='viridis', s=20, zorder=4, label='Points')
            
            # Mark start and end
            ax2.scatter(traj_arr[0, 0], traj_arr[0, 1], c='green', s=150, 
                       marker='o', zorder=5, label='Start', edgecolors='white', linewidths=2)
            ax2.scatter(traj_arr[-1, 0], traj_arr[-1, 1], c='red', s=150, 
                       marker='s', zorder=5, label='End', edgecolors='white', linewidths=2)
            
            # Set axis limits based on trajectory and arena
            if self.arena_detector.is_calibrated:
                arena_w, arena_h = self.arena_detector.arena_dimensions
                ax2.set_xlim(-50, arena_w + 50)
                ax2.set_ylim(-50, arena_h + 50)
        else:
            ax2.text(0.5, 0.5, 'No trajectory recorded', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12, color='gray')
        
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.invert_yaxis()  # Match image coordinates
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.suptitle(f'Arena Tracker Results - {len(path_points)} trajectory points', 
                    fontsize=12, y=1.02)
        
        print("✅ Plots generated. Close the plot window to exit.")
        plt.show()
    
    def send_data_to_robot(self):
        """Sends arena and path data to robot"""
        if not self.nxt_comm.is_connected:
            return
        
        if not self.nxt_comm.should_send():
            return
        
        # Send arena data
        if self.arena_detector.is_calibrated:
            corners = self.arena_detector.corners_flat.tolist()
            dims = self.arena_detector.arena_dimensions
            self.nxt_comm.send_arena_data(corners, dims)
        
        # Send path data
        path_points = list(self.robot_tracker.trajectory_points_flat)
        current_pos = self.robot_tracker.current_position_flat
        if path_points or current_pos:
            self.nxt_comm.send_path_data(path_points, current_pos)
        
        print(f"📤 Data sent to robot (path points: {len(path_points)})")
    
    def run(self):
        """Main run loop"""
        if not self.initialize_camera():
            return
        
        # Try to connect Bluetooth (non-blocking, continue even if fails)
        self.initialize_bluetooth()
        
        # Run manual calibration at startup
        print("\n🎯 Starting manual arena calibration...")
        print("   Position the camera to see the entire arena.")
        
        if not self.run_manual_calibration():
            print("\n⚠️ Calibration skipped. You can calibrate later with 'c' key.")
        
        self.running = True
        self.show_plots_on_exit = False  # Flag to show plots on ESC
        print("\n🚀 System started! Tracking robot...")
        print("📌 The robot will be tracked automatically.")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Frame grab failed")
                continue
            
            # Resize frame for processing
            frame = cv2.resize(frame, (0, 0), 
                              fx=self.config.DISPLAY_SCALE, 
                              fy=self.config.DISPLAY_SCALE)
            
            # Update robot tracker
            self.robot_tracker.update(frame, self.arena_detector)
            
            # Draw arena if calibrated
            if self.arena_detector.is_calibrated:
                frame = self.arena_detector.draw_arena(frame)
            
            # Draw robot trajectory
            frame = self.robot_tracker.draw_trajectory(frame)
            
            # Create side panel with UI and 2D map
            h, w = frame.shape[:2]
            side_panel = self.draw_side_panel(h)
            
            # Combine camera view with side panel (panel on left)
            combined = np.hstack([side_panel, frame])
            
            # Send data to robot periodically
            self.send_data_to_robot()
            
            # Display
            cv2.imshow("Arena Tracker System", combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # ESC key - quit and show plots
            if key == 27:
                print("\n👋 ESC pressed - Quitting and showing plots...")
                self.show_plots_on_exit = True
                self.running = False
            
            elif key == ord('q'):
                print("👋 Quitting...")
                self.running = False
            
            elif key == ord('c'):
                print("📐 Re-calibrating arena...")
                # Run manual calibration again
                if self.run_manual_calibration():
                    # Reset tracker when recalibrating
                    self.robot_tracker = RobotTracker(self.config)
                    print("✅ Recalibration complete, tracker reset.")
            
            elif key == ord('r'):
                print("🔄 Resetting tracker...")
                self.robot_tracker = RobotTracker(self.config)
            
            elif key == ord('b'):
                print("📡 Reconnecting Bluetooth...")
                self.nxt_comm.disconnect()
                self.initialize_bluetooth()
            
            elif key == ord('s'):
                # Save screenshot
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"💾 Screenshot saved: {filename}")
        
        # Show final plots if ESC was pressed
        if self.show_plots_on_exit:
            self.show_final_plots()
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        self.nxt_comm.disconnect()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("       ARENA TRACKER SYSTEM")
    print("=" * 60)
    print()
    print("This system will:")
    print("1. Ask you to click arena corners for calibration")
    print("2. Apply homography to create a 2D map view")
    print("3. Track robot position in real-time")
    print("4. Send arena & path data to NXT robot via Bluetooth")
    print()
    
    # Create custom config if needed
    config = Config()
    
    # You can modify these settings:
    # config.CAMERA_INDEX = 0  # Change camera index
    # config.NXT_ADDRESS = "XX:XX:XX:XX:XX:XX"  # Your NXT address
    # config.SEND_INTERVAL = 3.0  # Send data every 3 seconds
    
    # Create and run system
    system = ArenaTrackerSystem(config)
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
