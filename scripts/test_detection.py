#!/usr/bin/env python3
"""
Quick test script to debug ChArUco detection issues.
Run this standalone to test if detection works.

Usage:
    python3 test_detection.py                    # Use webcam
    python3 test_detection.py --image photo.jpg  # Use image file
    python3 test_detection.py --ros-topic /camera/rear_mid/image_raw  # Use ROS topic
"""

import cv2
import numpy as np
import argparse
import sys
import os

def test_opencv_aruco():
    """Test if OpenCV ArUco module is available."""
    print("Testing OpenCV ArUco module...")
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        print(f"  ✓ OpenCV version: {cv2.__version__}")
        print(f"  ✓ ArUco module available")
        return True
    except AttributeError as e:
        print(f"  ✗ ArUco module NOT available: {e}")
        print("  Install with: pip install opencv-contrib-python")
        return False

def create_detector(config_path=None):
    """Create ChArUco detector matching our config."""
    # Try to load from config file, fallback to defaults
    if config_path is None:
        # Try to find config file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'config', 'charuco_board.yaml')
    
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            squares_x = config.get('squares_x', 10)
            squares_y = config.get('squares_y', 14)
            square_length = config.get('square_length', 0.050)
            marker_length = config.get('marker_length', 0.037)
            dict_name = config.get('aruco_dictionary', 'DICT_6X6_250')
            print(f"  Loaded config: {squares_x}x{squares_y}, {square_length*1000:.0f}mm squares")
        except Exception as e:
            print(f"  Warning: Could not load config ({e}), using defaults")
            squares_x, squares_y = 10, 14
            square_length, marker_length = 0.050, 0.037
            dict_name = 'DICT_6X6_250'
    else:
        # Defaults (must match charuco_board.yaml if config not found)
        squares_x, squares_y = 10, 14
        square_length, marker_length = 0.050, 0.037
        dict_name = 'DICT_6X6_250'
    
    # Map dictionary name to OpenCV constant
    ARUCO_DICT_MAP = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    }
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP.get(dict_name, cv2.aruco.DICT_6X6_250))
    
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        dictionary
    )
    
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    return charuco_detector, dictionary

def detect_and_draw(image, charuco_detector, dictionary):
    """Detect ChArUco and draw results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        charuco_detector.detectBoard(gray)
    
    # Draw results
    vis = image.copy()
    
    # Draw ArUco markers (blue)
    if marker_corners is not None and len(marker_corners) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        num_markers = len(marker_corners)
    else:
        num_markers = 0
    
    # Draw ChArUco corners (green)
    if charuco_corners is not None and len(charuco_corners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
        num_corners = len(charuco_corners)
    else:
        num_corners = 0
    
    # Add status text
    status = f"Markers: {num_markers} | Corners: {num_corners}"
    cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    if num_markers == 0:
        cv2.putText(vis, "NO MARKERS DETECTED - Check lighting/board", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif num_corners == 0:
        cv2.putText(vis, "Markers found but no corners - Need more markers visible", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    return vis, num_markers, num_corners

def test_with_webcam(charuco_detector, dictionary):
    """Test detection with webcam."""
    print("\nOpening webcam (press 'q' to quit)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("  ✗ Could not open webcam")
        return
    
    cv2.namedWindow('ChArUco Detection Test', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        vis, num_markers, num_corners = detect_and_draw(frame, charuco_detector, dictionary)
        cv2.imshow('ChArUco Detection Test', vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_image(image_path, charuco_detector, dictionary):
    """Test detection with image file."""
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"  ✗ Could not load image: {image_path}")
        return
    
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    
    vis, num_markers, num_corners = detect_and_draw(image, charuco_detector, dictionary)
    
    print(f"  ArUco markers detected: {num_markers}")
    print(f"  ChArUco corners detected: {num_corners}")
    
    if num_markers == 0:
        print("\n  ⚠ NO MARKERS DETECTED!")
        print("  Possible causes:")
        print("    - Wrong ArUco dictionary (we use DICT_6X6_250)")
        print("    - Board not visible in image")
        print("    - Poor lighting or image quality")
        print("    - Board is too small in the image")
    
    cv2.namedWindow('ChArUco Detection Test', cv2.WINDOW_NORMAL)
    cv2.imshow('ChArUco Detection Test', vis)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_with_ros_topic(topic, charuco_detector, dictionary):
    """Test detection with ROS topic."""
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ImportError:
        print("  ✗ ROS2 not available. Run with --image instead.")
        return
    
    print(f"\nSubscribing to: {topic}")
    print("Press 'q' in window to quit...")
    
    rclpy.init()
    bridge = CvBridge()
    
    class TestNode(Node):
        def __init__(self):
            super().__init__('detection_test')
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.sub = self.create_subscription(Image, topic, self.callback, qos)
            self.got_image = False
            
        def callback(self, msg):
            self.got_image = True
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                vis, num_markers, num_corners = detect_and_draw(frame, charuco_detector, dictionary)
                cv2.imshow('ChArUco Detection Test', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rclpy.shutdown()
            except Exception as e:
                print(f"Error: {e}")
    
    cv2.namedWindow('ChArUco Detection Test', cv2.WINDOW_NORMAL)
    node = TestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description='Test ChArUco detection')
    parser.add_argument('--image', '-i', type=str, help='Test with image file')
    parser.add_argument('--ros-topic', '-r', type=str, help='Test with ROS topic')
    parser.add_argument('--webcam', '-w', action='store_true', help='Test with webcam')
    args = parser.parse_args()
    
    print("="*60)
    print("ChArUco Detection Test")
    print("="*60)
    
    # Test OpenCV
    if not test_opencv_aruco():
        sys.exit(1)
    
    # Create detector
    print("\nCreating detector from config...")
    charuco_detector, dictionary = create_detector()
    print("  ✓ Detector created")
    
    # Run appropriate test
    if args.image:
        test_with_image(args.image, charuco_detector, dictionary)
    elif args.ros_topic:
        test_with_ros_topic(args.ros_topic, charuco_detector, dictionary)
    else:
        # Default to webcam
        test_with_webcam(charuco_detector, dictionary)

if __name__ == '__main__':
    main()
