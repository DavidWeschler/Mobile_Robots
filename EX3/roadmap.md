# Project Roadmap: Mobile Robots (Differential Drive + Visual SLAM)

[cite_start]This roadmap is designed to secure the base grade (70 points) first by completing **Option 1**, and then attempt the bonus (up to 40 points) using **Option 6**[cite: 25, 36].

---

## STAGE 1: The Core Robot (Option 1)

**Goal:** Build a fully functional differential drive robot with odometry and data logging.
[cite_start]**Target Grade:** 70 Points[cite: 26].

### Phase 1.1: Hardware Construction

- [ ] **Chassis Design:** Build a robot with two driving wheels and a passive caster wheel (Differential Drive).
  - [cite_start]_Note:_ You must use the components available (Lego Mindstorms)[cite: 1].
  - [cite_start]_Requirement:_ The robot generally requires 3 motors (2 for drive, 1 auxiliary/turret)[cite: 6].
- [ ] **Measurements:** Measure and record the specific dimensions required for odometry:
  - **$R$**: Radius of the wheels.
  - **$L$**: Wheelbase (distance between the two driving wheels).
- [cite_start][ ] **Sensor Integration:** Install the compulsory sensors[cite: 6, 15]:
  - [cite_start][ ] **Distance Sensor:** 1 unit (Ultrasonic/IR), facing your chosen direction[cite: 15].
  - [cite_start][ ] **Touch Sensors:** 2 units (one front bumper, one rear bumper)[cite: 15].
  - [cite_start][ ] **Light Sensor:** 1 unit (required by general spec)[cite: 6].

### Phase 1.2: Embedded Software (Robot Side)

- [ ] **Basic Movement:** Implement functions for `DriveForward`, `TurnLeft`, `TurnRight`.
- [cite_start][ ] **Safety Logic:** Implement the required collision behaviors[cite: 17]:
  - [cite_start]Front Collision $\rightarrow$ Stop, Reverse, Turn[cite: 15, 17].
  - [cite_start]Rear Collision $\rightarrow$ Stop, Drive Forward[cite: 15, 17].
- [cite_start][ ] **Odometry Implementation:** Program the kinematic equations to update $(x, y, \theta)$ in the robot's loop[cite: 18].
  - **Formulas:**
    $$
    \Delta d = \frac{\Delta d_R + \Delta d_L}{2}
    $$
    $$
    \Delta \theta = \frac{\Delta d_R - \Delta d_L}{L}
    $$
    $$
    x_{new} = x_{old} + \Delta d \cdot \cos(\theta_{old} + \frac{\Delta \theta}{2})
    $$
    $$
    y_{new} = y_{old} + \Delta d \cdot \sin(\theta_{old} + \frac{\Delta \theta}{2})
    $$
- [cite_start][ ] **Display:** Show $(x, y)$ coordinates and cumulative path on the LCD screen in real-time[cite: 19].

### Phase 1.3: Data Logging

- [cite_start][ ] **Logging:** Create a mechanism to log the data to a file or transmit via Bluetooth/USB[cite: 23].
  - Data points: Timestamp, Calculated $x$, Calculated $y$, Sensor Readings.
- [cite_start][ ] **Auto-Stop:** Ensure the robot stops automatically after a set time (max 10 minutes)[cite: 22].

---

## STAGE 2: The Camera Upgrade (Option 6)

**Goal:** Use an external camera to correct odometry drift and generate a rectified map.
[cite_start]**Target Grade:** +40 Bonus Points[cite: 36].

### Phase 2.1: Vision System Setup

- [ ] **Camera Mounting:** Fix a camera (webcam/phone) high above the arena. [cite_start]It must be static[cite: 38].
- [ ] **Marker:** Add a distinct visual marker (e.g., a colored circle) to the top of your robot for easy tracking.
- [cite_start][ ] **Calibration Image:** Capture the empty arena to detect the static wall locations[cite: 39].

### Phase 2.2: Dual Data Collection

- [ ] **The Experiment:** Run the robot in the arena while simultaneously recording:
  1.  **Robot Log:** Internal odometry $(x, y)$ from Stage 1.
  2.  **Camera Feed:** Video of the robot moving.
- [cite_start][ ] **Synchronization:** Ensure you can align the robot's timestamps with the video's timeline[cite: 42].

### Phase 2.3: Fusion & SLAM (PC Software)

- [cite_start][ ] **Visual Tracking:** Write a script (Python/OpenCV) to extract the robot's position $(u, v)$ in pixels from the video frames[cite: 42].
- [ ] **Homography Calculation:**
  - [cite_start]Match the robot's calculated path points $(x, y)$ with the camera's observed points $(u, v)$[cite: 42].
  - [cite_start]Calculate the Homography Matrix $H$ that maps the image plane to the ground plane[cite: 42].
- [ ] **Map Generation:**
  - [cite_start]Apply $H$ to the wall pixels detected in Phase 2.1[cite: 42].
  - Generate a top-down metric map showing the true shape of the arena.
  - [cite_start]Overlay the robot's calculated path vs. the true path[cite: 43].

---

## Deliverables

**Due Date:** Lab Check on Friday 30.1.26 | [cite_start]Report on the following Tuesday[cite: 46].

### [cite_start]1. The Report [cite: 48]

- [cite_start][ ] **Robot Description:** Detailed explanation of the differential drive build[cite: 50].
- [cite_start][ ] **Odometry:** Explanation of the math used in Stage 1[cite: 50].
- [cite_start][ ] **SLAM Strategy:** Detailed explanation of the Homography fusion in Stage 2[cite: 51].
- [cite_start][ ] **Results:** Present the final map and discuss the accuracy[cite: 52].
- [cite_start][ ] **Media:** Link to a shared folder with photos and videos of the experiment[cite: 54].
- [cite_start][ ] **References:** List of sources[cite: 56].

### [cite_start]2. The Code [cite: 57]

- [ ] Complete source code for the Robot and the PC software.
- [ ] `readme.txt` explaining installation and operation.
