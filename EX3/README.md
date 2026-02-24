---
title: Students
---

## Students

- **David Weschler** (209736578)
- **Guy Danin** (205372105)
- **Benjamin Rosin** (211426598)

# SLAM System - End-to-End Workflow Guide

This document explains how to run the complete SLAM (Simultaneous Localization and Mapping) system for mobile robot arena navigation and mapping.

## System Overview

The SLAM system integrates robot odometry with computer vision to create accurate maps and trajectories. It consists of four main stages:

1. **Camera Calibration** - Set up and calibrate the camera
2. **Robot Navigation** - Run the robot and track it in real-time
3. **Optimization & Loop Closure** - Process odometry data and refine the map
4. **Analysis & Visualization** - Compare results and generate accuracy reports

---

## Prerequisites

### Hardware

- NXT Mobile Robot with:
  - Odometry encoders (wheels)
  - Ultrasonic distance sensor
  - Light sensor for anchor point detection
  - Touch sensor
- Smartphone camera on tripod positioned above the arena
- Arena with 5 marked corner points

### Software

- Python 3.8+ with packages:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `opencv-python` (cv2)
- NXC Compiler for robot code
- USB cable for robot programming

### File Structure

```
EX3/
├── slam_odometry.nxc              # Robot firmware
├── arena_tracker_system.py         # Camera tracking system
├── spring_clusre.py                # Optimization & loop closure
├── second_step.py                  # Analysis & visualization
├── requirements.txt                # Python dependencies
└── txt_files/                      # Data storage folder
    ├── clean_walls.txt             # Output: Cleaned wall segments
    ├── robot_path.txt              # Output: Optimized robot trajectory
    ├── trajectory.txt              # Output: Camera-tracked trajectory
    └── [input_data_files]
```

---

## Step 1: Camera Preparation & Calibration

### Setup

1. **Position the camera**: Mount the smartphone camera on a tripod above the arena center, ensuring all 5 corner points are visible in the frame with good lighting.

2. **Start the arena tracker system**:

   ```bash
   python arena_tracker_system.py
   ```

3. **Manual Corner Calibration**: When the script starts, a live camera window will appear. You must manually click on the 5 arena corners in order:
   - Top-Left
   - Top-Right
   - Bottom-Right
   - Bottom-Inner (inner corner)
   - Left-Inner (inner corner)

   **Important**: Click the corners in the exact sequence shown. This defines the homography matrix that maps camera pixels to real-world coordinates (cm).

4. **Verify calibration**: The script will display overlay lines on the arena corners to confirm calibration is correct.

### Output

- **homography matrix** is computed and stored in memory
- **trajectory.txt** will be created to log camera-tracked positions during the run

---

## Step 2: Robot Navigation & Real-Time Tracking

### Compile Robot Code

1. Open the NXC compiler (NXT-G or similar)
2. Compile `slam_odometry.nxc` and upload to the robot:
   ```
   File → Compile & Download
   ```

### Run the Robot

1. **Start the camera tracking** (if not already running from Step 1):

   ```bash
   python arena_tracker_system.py
   ```

2. **Insert robot into arena** and press the robot's "Start" button to begin autonomous navigation.

3. **What happens**:
   - The robot follows walls using PID control
   - Ultrasonic sensor measures wall distances every 5 loop iterations
   - Light sensor detects black anchor points along walls
   - Odometry encoders track wheel rotations (X, Y, Theta position)
   - Every 5th data point is logged to memory
   - Camera simultaneously tracks the robot's real position

4. **Robot stops automatically** after traveling ~11 meters (>1 full lap) and saves data to `TRACK_LOG.TXT` on the brick.

5. **Download data from robot**:
   - Connect robot via USB
   - Download `TRACK_LOG.TXT` to the `txt_files/` folder
   - Rename it to match input expectations if needed

### Output Files

- **TRACK_LOG.TXT** - Raw odometry data: X, Y, Theta, Distance, Status (5 columns)
- **trajectory.txt** - Camera-tracked robot positions (computed by arena_tracker_system.py)

---

## Step 3: Optimization & Loop Closure

### Run the Optimization Script

```bash
python spring_clusre.py
```

### What This Script Does

1. **Reads raw odometry** from `TRACK_LOG.TXT`
2. **Detects anchor points** - Identifies black markers on walls using light sensor data
3. **Matches anchor points** - Correlates repeated detections across the trajectory to close the loop
4. **Applies spring model optimization** - Distributes drift error along the entire path
5. **Extracts walls with RANSAC** - Identifies straight wall segments and removes outliers
6. **Generates clean output**:
   - Saves wall segments with coordinates and lengths
   - Generates corrected robot trajectory

### Key Parameters (adjustable in script)

- `tolerance` - Clustering distance for anchor point matching
- `spring_iterations` - Number of optimization passes
- `ransac_threshold` - Distance tolerance for wall fitting
- `min_wall_length` - Minimum wall segment length to include

### Output Files

- **clean_walls.txt** - Formatted wall segments: X_Start, Y_Start, X_End, Y_End, Length_cm
- **robot_path.txt** - Optimized odometry trajectory: X_cm, Y_cm (with corrections applied)

---

## Step 4: Analysis & Visualization

### Run the Analysis Script

```bash
python second_step.py
```

### What This Script Does

#### Part 1: Arena Corner Comparison

- Loads cleaned wall segments and extracts arena vertices
- Clusters endpoint coordinates to find the 5 corner points
- **Aligns measured arena** to the true arena reference using:
  - Translation
  - Rotation
  - Optional mirroring (if needed)
- **Calculates edge lengths** for each wall
- **Calculates corner angles** at each vertex
- **Reports errors**:
  - Mean edge length error (cm and %)
  - Mean angle error (degrees and %)

#### Part 2: Trajectory Comparison

- Loads robot odometry trajectory (from `robot_path.txt`)
- Loads camera ground-truth trajectory (from `trajectory.txt`)
- **Applies arena transform** to odometry trajectory to align with ground-truth coordinates
- **Computes point-wise errors** using nearest-neighbor distances
- **Reports trajectory metrics**:
  - Mean error
  - Median error
  - Max error
  - Standard deviation
  - RMSE (Root Mean Square Error)

#### Visualization

Two figures are generated:

**Figure 1: Combined View**

- True arena (green outline with vertices labeled T0-T4)
- Measured arena (red dashed outline with vertices labeled M0-M4)
- Robot odometry trajectory (blue)
- Camera trajectory (orange)
- Wall lengths displayed on each segment

**Figure 2: Detailed 4-Subplot Analysis**

- **Top-Left**: Arena shape comparison with wall lengths
- **Top-Right**: Error distribution histogram
- **Bottom-Left**: Aligned trajectories overlay
- **Bottom-Right**: Summary table with all metrics

### Output

- Console printout of all metrics
- Two interactive matplotlib figures
- All data ready for report writing

---

## Complete End-to-End Workflow

### Quick Start Checklist

```
□ Step 1: CAMERA CALIBRATION
  □ Position camera above arena
  □ Run: python arena_tracker_system.py
  □ Click 5 corners when prompted
  □ Verify calibration overlay looks correct

□ Step 2: ROBOT EXECUTION
  □ Compile slam_odometry.nxc to robot
  □ Insert robot into arena
  □ Press robot start button
  □ Wait for robot to complete lap (~2-3 minutes)
  □ Download TRACK_LOG.TXT from robot to txt_files/

□ Step 3: OPTIMIZATION
  □ Run: python spring_clusre.py
  □ Wait for spring model convergence
  □ Check outputs: clean_walls.txt, robot_path.txt

□ Step 4: ANALYSIS
  □ Run: python second_step.py
  □ Review console output for metrics
  □ Examine generated plots
  □ Save results/figures for report
```

---

## File Reference

| File                      | Purpose                            | Input/Output |
| ------------------------- | ---------------------------------- | ------------ |
| `slam_odometry.nxc`       | Robot firmware with PID + odometry | Input        |
| `arena_tracker_system.py` | Camera calibration & tracking      | Input        |
| `spring_clusre.py`        | Optimization & loop closure        | Input        |
| `second_step.py`          | Analysis & visualization           | Input        |
| `TRACK_LOG.TXT`           | Raw robot odometry                 | Intermediate |
| `clean_walls.txt`         | Extracted wall segments            | Output       |
| `robot_path.txt`          | Optimized trajectory               | Output       |
| `trajectory.txt`          | Camera ground-truth                | Output       |
