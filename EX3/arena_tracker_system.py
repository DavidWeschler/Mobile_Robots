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
    
    # Arena Detection Settings
    MIN_ARENA_AREA: int = 5000  # Minimum contour area to be considered arena
    POLYGON_EPSILON: float = 0.015  # Polygon approximation accuracy (1.5%)
    
    # Robot Tracking Settings
    BG_HISTORY: int = 500  # Background subtractor history
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
# ARENA DETECTOR CLASS
# =============================================================================

class ArenaDetector:
    """Detects arena boundaries using red tape markers"""
    
    def __init__(self, config: Config):
        self.config = config
        self.corners_original = None
        self.corners_flat = None
        self.homography_matrix = None
        self.arena_dimensions = None
        self.walls = []
        self.is_calibrated = False
    
    def get_red_mask(self, img_hsv: np.ndarray) -> np.ndarray:
        """Isolates red tape from HSV image"""
        # Red wraps around in HSV, so we need two ranges
        lower1, upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower2, upper2 = np.array([160, 100, 100]), np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower1, upper1)
        mask2 = cv2.inRange(img_hsv, lower2, upper2)
        return mask1 + mask2
    
    def order_rect_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders 4 points of bounding box as (TL, TR, BR, BL)"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect
    
    def detect_arena(self, frame: np.ndarray) -> bool:
        """
        Detects arena from a frame. Returns True if successful.
        Sets corners, walls, and homography matrix.
        """
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get red mask
            mask = self.get_red_mask(hsv)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
            
            # Get largest contour (arena)
            arena_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(arena_contour) < self.config.MIN_ARENA_AREA:
                return False
            
            # Approximate polygon to get corners
            epsilon = self.config.POLYGON_EPSILON * cv2.arcLength(arena_contour, True)
            polygon_corners = cv2.approxPolyDP(arena_contour, epsilon, True)
            
            num_corners = len(polygon_corners)
            if num_corners < 3:
                return False
            
            self.corners_original = polygon_corners
            
            # Calculate perspective transform using bounding rectangle
            rect = cv2.minAreaRect(arena_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            src_rect_pts = self.order_rect_points(box.reshape(4, 2).astype("float32"))
            
            # Calculate dimensions
            (tl, tr, br, bl) = src_rect_pts
            width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
            
            self.arena_dimensions = (width, height)
            
            dst_rect_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]], dtype="float32")
            
            # Calculate homography matrix
            self.homography_matrix = cv2.getPerspectiveTransform(src_rect_pts, dst_rect_pts)
            
            # Transform corners to flat 2D space
            pts_original = polygon_corners.reshape(-1, 1, 2).astype("float32")
            pts_flat = cv2.perspectiveTransform(pts_original, self.homography_matrix)
            self.corners_flat = pts_flat.reshape(-1, 2)
            
            # Extract walls as line segments
            self.walls = []
            for i in range(len(self.corners_flat)):
                p1 = self.corners_flat[i]
                p2 = self.corners_flat[(i + 1) % len(self.corners_flat)]
                self.walls.append((p1.tolist(), p2.tolist()))
            
            self.is_calibrated = True
            print(f"✅ Arena detected with {num_corners} corners!")
            return True
            
        except Exception as e:
            print(f"❌ Arena detection error: {e}")
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
        
        # Draw polygon
        cv2.drawContours(frame, [self.corners_original], -1, (0, 255, 0), 2)
        
        # Draw corner points
        for i, pt in enumerate(self.corners_original):
            x, y = pt[0]
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
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draws UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Status panel background
        cv2.rectangle(frame, (5, 5), (300, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 130), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "Arena Tracker System", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Arena status
        arena_status = "CALIBRATED" if self.arena_detector.is_calibrated else "NOT CALIBRATED"
        arena_color = (0, 255, 0) if self.arena_detector.is_calibrated else (0, 0, 255)
        cv2.putText(frame, f"Arena: {arena_status}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, arena_color, 1)
        
        # Tracking status
        if not self.robot_tracker.is_tracking:
            progress = self.robot_tracker.get_warmup_progress()
            cv2.putText(frame, f"Warmup: {int(progress * 100)}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            track_status = "TRACKING" if self.robot_tracker.current_position else "SEARCHING"
            track_color = (0, 255, 0) if self.robot_tracker.current_position else (255, 165, 0)
            cv2.putText(frame, f"Robot: {track_status}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)
        
        # Bluetooth status
        bt_status = "CONNECTED" if self.nxt_comm.is_connected else "DISCONNECTED"
        bt_color = (0, 255, 0) if self.nxt_comm.is_connected else (0, 0, 255)
        cv2.putText(frame, f"Bluetooth: {bt_status}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bt_color, 1)
        
        # Path points count
        path_count = len(self.robot_tracker.trajectory_points)
        cv2.putText(frame, f"Path points: {path_count}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'c' to calibrate arena, 'q' to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'r' to reset tracking, 'b' to reconnect BT", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
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
        
        # Title
        cv2.putText(map_img, "2D Map View", (10, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return map_img
    
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
        
        self.running = True
        print("\n🚀 System started! Point camera at the arena.")
        print("📌 Make sure the arena has RED tape borders!")
        
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
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            # Create 2D map view
            map_view = self.draw_2d_map((300, 300))
            
            # Combine main view and map view
            h, w = frame.shape[:2]
            # Place map in bottom-right corner
            frame[h-300:h, w-300:w] = map_view
            
            # Send data to robot periodically
            self.send_data_to_robot()
            
            # Display
            cv2.imshow("Arena Tracker System", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Quitting...")
                self.running = False
            
            elif key == ord('c'):
                print("📐 Calibrating arena...")
                if self.arena_detector.detect_arena(frame):
                    self.nxt_comm.send_status("Arena OK")
                else:
                    print("❌ Calibration failed. Make sure red tape is visible.")
            
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
    print("1. Detect arena boundaries (red tape)")
    print("2. Track robot position in real-time")
    print("3. Send data to NXT robot via Bluetooth")
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
