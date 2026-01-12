"""
ROS2 node for collecting synchronized images from multiple cameras for calibration.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import yaml
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from threading import Lock
import queue

from .charuco_detector import CharucoDetector
from .utils import load_camera_config, load_charuco_config


@dataclass
class CameraBuffer:
    """Buffer for storing the latest image from a camera."""
    image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: str = ""


class ImageCollectorNode(Node):
    """
    ROS2 node that subscribes to camera image topics and collects
    synchronized frame pairs for calibration.
    """
    
    def __init__(self):
        super().__init__('image_collector')
        
        # Declare parameters
        self.declare_parameter('cameras_config', '')
        self.declare_parameter('charuco_config', '')
        self.declare_parameter('output_dir', '/tmp/calibration_images')
        self.declare_parameter('sync_tolerance_ms', 100.0)  # Time sync tolerance
        self.declare_parameter('min_corners', 8)
        self.declare_parameter('auto_capture', False)
        self.declare_parameter('auto_capture_interval', 2.0)  # seconds
        
        # Get parameters
        cameras_config_path = self.get_parameter('cameras_config').value
        charuco_config_path = self.get_parameter('charuco_config').value
        self.output_dir = self.get_parameter('output_dir').value
        self.sync_tolerance = self.get_parameter('sync_tolerance_ms').value / 1000.0
        self.min_corners = self.get_parameter('min_corners').value
        self.auto_capture = self.get_parameter('auto_capture').value
        self.auto_capture_interval = self.get_parameter('auto_capture_interval').value
        
        # Load configurations
        if not cameras_config_path or not charuco_config_path:
            self.get_logger().error("cameras_config and charuco_config parameters are required")
            raise ValueError("Missing configuration paths")
        
        self.cameras_config = load_camera_config(cameras_config_path)
        charuco_config = load_charuco_config(charuco_config_path)
        
        # Initialize ChArUco detector
        self.detector = CharucoDetector(charuco_config)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Camera buffers and subscribers
        self.buffers: Dict[str, CameraBuffer] = {}
        self.subscribers: Dict[str, any] = {}
        self.buffer_lock = Lock()
        
        # Capture state
        self.capture_count = 0
        self.last_auto_capture = 0.0
        self.capture_queue = queue.Queue()
        
        # Calibration pairs from config
        self.calibration_pairs = self.cameras_config.get('calibration_pairs', [])
        
        # Set up QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to all camera topics
        for cam_name, cam_config in self.cameras_config['cameras'].items():
            topic = cam_config['image_topic']
            self.buffers[cam_name] = CameraBuffer()
            
            self.subscribers[cam_name] = self.create_subscription(
                Image,
                topic,
                lambda msg, name=cam_name: self.image_callback(msg, name),
                qos
            )
            self.get_logger().info(f"Subscribed to {topic} for camera '{cam_name}'")
        
        # Timer for processing and display
        self.display_timer = self.create_timer(0.1, self.display_callback)
        
        # Create windows for visualization
        cv2.namedWindow('Calibration View', cv2.WINDOW_NORMAL)
        
        self.get_logger().info(f"Image collector initialized. Output: {self.output_dir}")
        self.get_logger().info("Press 'c' to capture, 'q' to quit, 's' to show stats")
    
    def image_callback(self, msg: Image, camera_name: str):
        """Handle incoming image messages."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.buffer_lock:
                self.buffers[camera_name].image = cv_image
                self.buffers[camera_name].timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.buffers[camera_name].frame_id = msg.header.frame_id
                
        except Exception as e:
            self.get_logger().error(f"Error processing image from {camera_name}: {e}")
    
    def display_callback(self):
        """Periodic callback for display and capture logic."""
        with self.buffer_lock:
            # Get images from first two cameras with overlapping FOV for display
            display_images = []
            cam_names = []
            
            # Show images from first calibration pair
            if self.calibration_pairs:
                pair = self.calibration_pairs[0]
                for cam_name in pair:
                    if cam_name in self.buffers and self.buffers[cam_name].image is not None:
                        img = self.buffers[cam_name].image.copy()
                        # Detect ChArUco and draw
                        det = self.detector.detect(img, draw_detections=True)
                        if det.image_with_detections is not None:
                            img = det.image_with_detections
                        
                        # Add camera name and corner count
                        cv2.putText(img, f"{cam_name} ({det.num_corners} corners)", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        display_images.append(img)
                        cam_names.append(cam_name)
        
        if display_images:
            # Resize for display
            display_h = 480
            resized = []
            for img in display_images:
                h, w = img.shape[:2]
                scale = display_h / h
                resized.append(cv2.resize(img, (int(w * scale), display_h)))
            
            # Concatenate horizontally
            if len(resized) >= 2:
                combined = np.hstack(resized[:2])
            else:
                combined = resized[0]
            
            cv2.imshow('Calibration View', combined)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            self.capture_all_pairs()
        elif key == ord('q'):
            self.get_logger().info("Quit requested")
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord('s'):
            self.print_statistics()
        
        # Auto capture logic
        if self.auto_capture:
            current_time = time.time()
            if current_time - self.last_auto_capture >= self.auto_capture_interval:
                self.capture_all_pairs()
                self.last_auto_capture = current_time
    
    def capture_all_pairs(self):
        """Capture images for all calibration pairs."""
        with self.buffer_lock:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            for pair in self.calibration_pairs:
                cam1, cam2 = pair
                
                # Check if both cameras have recent images
                buf1 = self.buffers.get(cam1)
                buf2 = self.buffers.get(cam2)
                
                if buf1 is None or buf1.image is None:
                    self.get_logger().warn(f"No image from {cam1}")
                    continue
                if buf2 is None or buf2.image is None:
                    self.get_logger().warn(f"No image from {cam2}")
                    continue
                
                # Check time sync
                time_diff = abs(buf1.timestamp - buf2.timestamp)
                if time_diff > self.sync_tolerance:
                    self.get_logger().warn(
                        f"Images not synchronized: {cam1}-{cam2} diff={time_diff:.3f}s"
                    )
                    continue
                
                # Detect ChArUco in both images
                det1 = self.detector.detect(buf1.image)
                det2 = self.detector.detect(buf2.image)
                
                if not det1.success or det1.num_corners < self.min_corners:
                    self.get_logger().info(
                        f"Insufficient corners in {cam1}: {det1.num_corners}"
                    )
                    continue
                    
                if not det2.success or det2.num_corners < self.min_corners:
                    self.get_logger().info(
                        f"Insufficient corners in {cam2}: {det2.num_corners}"
                    )
                    continue
                
                # Check for common corners
                common_ids = np.intersect1d(det1.charuco_ids, det2.charuco_ids)
                if len(common_ids) < self.min_corners:
                    self.get_logger().info(
                        f"Insufficient common corners: {len(common_ids)}"
                    )
                    continue
                
                # Save images
                pair_dir = os.path.join(self.output_dir, f"{cam1}_{cam2}")
                os.makedirs(pair_dir, exist_ok=True)
                
                img1_path = os.path.join(pair_dir, f"{timestamp}_{cam1}.png")
                img2_path = os.path.join(pair_dir, f"{timestamp}_{cam2}.png")
                
                cv2.imwrite(img1_path, buf1.image)
                cv2.imwrite(img2_path, buf2.image)
                
                self.capture_count += 1
                self.get_logger().info(
                    f"✓ Captured {cam1}-{cam2} ({det1.num_corners}/{det2.num_corners} corners, "
                    f"{len(common_ids)} common) - Total: {self.capture_count}"
                )
    
    def print_statistics(self):
        """Print capture statistics."""
        self.get_logger().info("=== Capture Statistics ===")
        self.get_logger().info(f"Total captures: {self.capture_count}")
        
        for pair in self.calibration_pairs:
            cam1, cam2 = pair
            pair_dir = os.path.join(self.output_dir, f"{cam1}_{cam2}")
            if os.path.exists(pair_dir):
                count = len([f for f in os.listdir(pair_dir) if f.endswith('.png')]) // 2
                self.get_logger().info(f"  {cam1} <-> {cam2}: {count} pairs")
            else:
                self.get_logger().info(f"  {cam1} <-> {cam2}: 0 pairs")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ImageCollectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
