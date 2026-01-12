# Camera Extrinsic Calibration

A ROS2 package for multi-camera extrinsic calibration using ChArUco boards. Designed for the tractor camera system with 7 cameras (5 surround fisheye cameras + ZED stereo pair) from Leipzig Universities Smart Farming Lab.

## Overview

This package provides tools to:
1. Generate a printable ChArUco calibration board (A1 size)
2. Collect synchronized calibration images from camera pairs
3. Compute extrinsic transforms between cameras
4. Visualize and export calibration results

## Camera Setup

The calibration is designed for the following camera configuration:

| Camera | Type | Resolution | Topic |
|--------|------|------------|-------|
| `rear_mid` (reference) | Fisheye | 1280×720 | `/camera/rear_mid/image_raw` |
| `rear_left` | Fisheye | 1280×720 | `/camera/rear_left/image_raw` |
| `rear_right` | Fisheye | 1280×720 | `/camera/rear_right/image_raw` |
| `side_left` | Fisheye | 1280×720 | `/camera/side_left/image_raw` |
| `side_right` | Fisheye | 1280×720 | `/camera/side_right/image_raw` |
| `zed_left` | Pinhole | 1104×621 | `/zed/zed_node/left_raw/image_raw_color` |
| `zed_right` | Pinhole | 1104×621 | `/zed/zed_node/right_raw/image_raw_color` |

### Calibration Chain

The extrinsic calibration propagates from `rear_mid` (reference) through overlapping camera pairs:

```
rear_mid (BASE)
├── rear_left
│   └── side_left
└── rear_right
    └── side_right
        └── zed_left
            └── zed_right
```

## Prerequisites

- ROS2 (Humble or later)
- Python 3.8+
- OpenCV with ArUco support (`opencv-contrib-python`)
- NumPy, SciPy, PyYAML
- ReportLab and Pillow (for PDF board generation)
- cv_bridge, image_transport

```bash
# Install Python dependencies
pip install opencv-contrib-python numpy scipy pyyaml reportlab pillow

# ROS2 dependencies
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-image-transport
```

## Installation

```bash
# Clone or copy the package to your ROS2 workspace
cd ~/ros2_ws/src
# (copy camera_extrinsic_calibration here)

# Build
cd ~/ros2_ws
colcon build --packages-select camera_extrinsic_calibration

# Source
source install/setup.bash
```

## Usage

### Step 1: Generate ChArUco Board

Generate a printable ChArUco board. **PDF output is recommended** as it ensures correct physical dimensions:

```bash
# Generate PDF for A1 paper (recommended, auto-detects paper size)
ros2 run camera_extrinsic_calibration generate_charuco_board.py \
    --config $(ros2 pkg prefix camera_extrinsic_calibration)/share/camera_extrinsic_calibration/config/charuco_board.yaml \
    --output charuco_board.pdf \
    --paper-size A1

# Or using config file (simpler)
ros2 run camera_extrinsic_calibration generate_charuco_board.py \
    --config $(ros2 pkg prefix camera_extrinsic_calibration)/share/camera_extrinsic_calibration/config/charuco_board.yaml \
    --output charuco_board.pdf

# Custom board parameters
ros2 run camera_extrinsic_calibration generate_charuco_board.py \
    --squares-x 8 \
    --squares-y 10 \
    --square-length 50.0 \
    --marker-length 37.0 \
    --paper-size A1 \
    --output board.pdf

# Generate PNG (less reliable for printing)
ros2 run camera_extrinsic_calibration generate_charuco_board.py \
    --output charuco_board.png \
    --paper-size A1
```

**Supported paper sizes:** A0, A1, A2, A3, A4, Letter, Legal, Tabloid

**Important printing instructions:**
1. **For PDF:** Print at **"Actual Size"** or **"100%"** scale (do NOT use "Fit to Page")
2. **For PNG:** Print at **100% scale** (no fit-to-page)
3. Use **matte paper** to avoid reflections
4. Mount on a **rigid, flat surface** (foam board recommended)
5. **Verify** the square size with a ruler (should match your config exactly)

### Step 2: Collect Calibration Images

Start the image collection node:

```bash
ros2 launch camera_extrinsic_calibration collect_images.launch.py \
    output_dir:=/path/to/save/images
```

**Controls:**
- Press `c` to capture images from all camera pairs
- Press `n` to show next camera pair in view
- Press `p` to show previous camera pair in view
- Press `1`-`9` to jump directly to pair 1-9
- Press `s` to show capture statistics
- Press `q` to quit

**Tips for good calibration:**
- Collect at least 15-20 image pairs per camera pair
- Move the board to different positions and orientations
- Ensure the board is visible in both cameras of each pair
- Cover different distances (0.5m - 3m from cameras)
- Include various angles (tilted board)

### Step 3: Compute Extrinsics

After collecting images, compute the extrinsic calibration:

```bash
# Using the tractor_multi_cam_publisher intrinsics
ros2 run camera_extrinsic_calibration compute_extrinsics.py \
    --cameras-config $(ros2 pkg prefix camera_extrinsic_calibration)/share/camera_extrinsic_calibration/config/cameras.yaml \
    --charuco-config $(ros2 pkg prefix camera_extrinsic_calibration)/share/camera_extrinsic_calibration/config/charuco_board.yaml \
    --images-dir /path/to/calibration_images \
    --intrinsics-dir $(ros2 pkg prefix tractor_multi_cam_publisher)/share/tractor_multi_cam_publisher/calibration \
    --output extrinsics_calibrated.yaml \
    --output-urdf extrinsics_calibrated.urdf
```

### Step 4: Visualize Results

Visualize the computed extrinsics:

```bash
ros2 run camera_extrinsic_calibration visualize_calibration.py \
    --extrinsics extrinsics_calibrated.yaml
```

## Configuration Files

### cameras.yaml

Defines camera topics, intrinsics files, and calibration pairs:

```yaml
reference_camera: "rear_mid"

cameras:
  rear_mid:
    image_topic: "/camera/rear_mid/image_raw"
    intrinsics_file: "camera_rear_mid.intrinsics.yaml"
    distortion_model: "fisheye"
    
calibration_pairs:
  - ["rear_mid", "rear_left"]
  - ["rear_mid", "rear_right"]
  # ... etc
```

### charuco_board.yaml

Defines the ChArUco board parameters:

```yaml
squares_x: 8          # Number of chessboard squares in X direction
squares_y: 10         # Number of chessboard squares in Y direction
square_length: 0.0692 # Square side length in meters (69.2mm)
marker_length: 0.05   # ArUco marker side length in meters (50mm)
aruco_dictionary: "DICT_6X6_250"

calibration:
  min_corners_per_frame: 9  # Adjust if you change grid size!
                            # Should be ~30-50% of typical visible corners
```

**Note:** If you change `squares_x` or `squares_y`, also adjust `min_corners_per_frame` proportionally. The total number of corners on the board is `(squares_x - 1) * (squares_y - 1)`.

## Output Format

The computed extrinsics are saved in the same format as your existing `extrinsics.yaml`:

```yaml
# Position xyz; Quaternions xyzw
# [ x_m, y_m, z_m, qx, qy, qz, qw]
rear_left:
  parent: "rear_mid"
  child: "rear_left"
  value: [0.123456, -0.234567, 0.345678, 0.1, 0.2, 0.3, 0.9]
```

## Troubleshooting

### ChArUco board not detected
- Ensure adequate lighting (avoid harsh shadows)
- Check board is flat and not warped
- Verify printed square size matches config (use a ruler!)
- Try adjusting detection parameters in `charuco_board.yaml`
- Ensure board is large enough in the image (at least 20-30% of frame)

### PDF generation fails
- Install required packages: `pip install reportlab pillow`
- Check that board dimensions fit on selected paper size
- Script will warn if board is too large for the paper

### Poor calibration results
- Collect more image pairs (20+ recommended)
- Ensure good coverage of different positions/angles
- Check intrinsics are accurate
- Verify cameras have overlapping FOV

### Time synchronization issues
- Increase `sync_tolerance_ms` parameter
- Ensure camera timestamps are accurate
- Use hardware sync if available

## File Structure

```
camera_extrinsic_calibration/
├── camera_extrinsic_calibration/
│   ├── __init__.py
│   ├── charuco_detector.py      # ChArUco detection
│   ├── calibration_solver.py    # Extrinsic computation
│   ├── image_collector.py       # ROS2 image collection node
│   └── utils.py                 # Helper functions
├── config/
│   ├── cameras.yaml             # Camera configuration
│   └── charuco_board.yaml       # Board configuration
├── launch/
│   ├── collect_images.launch.py
│   └── calibrate.launch.py
├── scripts/
│   ├── generate_charuco_board.py
│   ├── collect_calibration_images.py
│   ├── compute_extrinsics.py
│   └── visualize_calibration.py
├── CMakeLists.txt
├── package.xml
└── README.md
```

## License

MIT License
