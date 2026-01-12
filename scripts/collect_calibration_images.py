#!/usr/bin/env python3
"""
ROS2 script to collect calibration images from multiple cameras.

This script subscribes to camera topics and collects synchronized images
when the ChArUco board is visible in overlapping camera pairs.

Usage:
    ros2 run camera_extrinsic_calibration collect_calibration_images.py \
        --ros-args \
        -p cameras_config:=/path/to/cameras.yaml \
        -p charuco_config:=/path/to/charuco_board.yaml \
        -p output_dir:=/path/to/output
"""

import rclpy
import sys
import os

# Add package to path for standalone execution
try:
    from camera_extrinsic_calibration.image_collector import main
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_extrinsic_calibration.image_collector import main


if __name__ == '__main__':
    main()
