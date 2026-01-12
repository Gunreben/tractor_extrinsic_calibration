"""
ChArUco board detection for camera calibration.
Supports both pinhole and fisheye camera models.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


# ArUco dictionary mapping
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


@dataclass
class DetectionResult:
    """Result of ChArUco detection on an image."""
    success: bool
    charuco_corners: Optional[np.ndarray] = None  # Detected corner positions in image
    charuco_ids: Optional[np.ndarray] = None      # IDs of detected corners
    marker_corners: Optional[List] = None         # ArUco marker corners
    marker_ids: Optional[np.ndarray] = None       # ArUco marker IDs
    num_corners: int = 0
    image_with_detections: Optional[np.ndarray] = None


class CharucoDetector:
    """
    ChArUco board detector with support for fisheye and pinhole cameras.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ChArUco detector.
        
        Args:
            config: ChArUco board configuration dictionary
        """
        self.squares_x = config['squares_x']
        self.squares_y = config['squares_y']
        self.square_length = config['square_length']
        self.marker_length = config['marker_length']
        
        # Get ArUco dictionary
        dict_name = config.get('aruco_dictionary', 'DICT_6X6_250')
        if dict_name not in ARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])
        
        # Create ChArUco board
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        
        # Configure detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        
        detection_cfg = config.get('detection', {})
        if detection_cfg:
            self._configure_detector_params(detection_cfg)
        
        # Create ArUco detector
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # Create CharucoDetector for refined corner detection
        charuco_params = cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.board, charuco_params, self.detector_params
        )
    
    def _configure_detector_params(self, cfg: Dict[str, Any]):
        """Configure ArUco detector parameters from config."""
        param_map = {
            'adaptive_thresh_win_size_min': 'adaptiveThreshWinSizeMin',
            'adaptive_thresh_win_size_max': 'adaptiveThreshWinSizeMax',
            'adaptive_thresh_win_size_step': 'adaptiveThreshWinSizeStep',
            'adaptive_thresh_constant': 'adaptiveThreshConstant',
            'min_marker_perimeter_rate': 'minMarkerPerimeterRate',
            'max_marker_perimeter_rate': 'maxMarkerPerimeterRate',
            'polygonal_approx_accuracy_rate': 'polygonalApproxAccuracyRate',
            'min_corner_distance_rate': 'minCornerDistanceRate',
            'min_distance_to_border': 'minDistanceToBorder',
            'min_marker_distance_rate': 'minMarkerDistanceRate',
            'corner_refinement_win_size': 'cornerRefinementWinSize',
            'corner_refinement_max_iterations': 'cornerRefinementMaxIterations',
            'corner_refinement_min_accuracy': 'cornerRefinementMinAccuracy',
        }
        
        for yaml_key, cv_attr in param_map.items():
            if yaml_key in cfg:
                setattr(self.detector_params, cv_attr, cfg[yaml_key])
        
        # Handle corner refinement method specially
        if 'corner_refinement_method' in cfg:
            method_map = {
                'CORNER_REFINE_NONE': cv2.aruco.CORNER_REFINE_NONE,
                'CORNER_REFINE_SUBPIX': cv2.aruco.CORNER_REFINE_SUBPIX,
                'CORNER_REFINE_CONTOUR': cv2.aruco.CORNER_REFINE_CONTOUR,
                'CORNER_REFINE_APRILTAG': cv2.aruco.CORNER_REFINE_APRILTAG,
            }
            method = cfg['corner_refinement_method']
            if method in method_map:
                self.detector_params.cornerRefinementMethod = method_map[method]
    
    def detect(self, image: np.ndarray, 
               camera_matrix: Optional[np.ndarray] = None,
               dist_coeffs: Optional[np.ndarray] = None,
               is_fisheye: bool = False,
               draw_detections: bool = False) -> DetectionResult:
        """
        Detect ChArUco board in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            camera_matrix: 3x3 camera intrinsic matrix (optional, for undistortion)
            dist_coeffs: Distortion coefficients (optional)
            is_fisheye: Whether to use fisheye undistortion model
            draw_detections: Whether to draw detections on the image
            
        Returns:
            DetectionResult with detection information
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect using the CharucoDetector
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            self.charuco_detector.detectBoard(gray)
        
        # Check if detection was successful
        if charuco_corners is None or len(charuco_corners) < 4:
            return DetectionResult(success=False)
        
        # Refine corners if camera parameters are available
        if camera_matrix is not None and dist_coeffs is not None and not is_fisheye:
            # For pinhole cameras, we can refine corners
            charuco_corners = cv2.cornerSubPix(
                gray, charuco_corners,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
        
        result = DetectionResult(
            success=True,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            marker_corners=marker_corners,
            marker_ids=marker_ids,
            num_corners=len(charuco_corners)
        )
        
        # Draw detections if requested
        if draw_detections:
            vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Draw ArUco markers
            if marker_corners is not None and len(marker_corners) > 0:
                cv2.aruco.drawDetectedMarkers(vis_image, marker_corners, marker_ids)
            
            # Draw ChArUco corners
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(vis_image, charuco_corners, charuco_ids)
            
            result.image_with_detections = vis_image
        
        return result
    
    def estimate_pose(self, charuco_corners: np.ndarray, charuco_ids: np.ndarray,
                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                      is_fisheye: bool = False) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Estimate the pose of the ChArUco board.
        
        Args:
            charuco_corners: Detected corner positions
            charuco_ids: IDs of detected corners
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            is_fisheye: Whether camera uses fisheye model
            
        Returns:
            Tuple of (success, rvec, tvec)
        """
        if charuco_corners is None or len(charuco_corners) < 4:
            return False, None, None
        
        # Get object points for detected corners
        obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
        
        if obj_points is None or len(obj_points) < 4:
            return False, None, None
        
        if is_fisheye:
            # For fisheye, we need to undistort points first, then use regular solvePnP
            undistorted = cv2.fisheye.undistortPoints(
                img_points.reshape(-1, 1, 2), 
                camera_matrix, 
                dist_coeffs
            )
            # Use identity camera matrix after undistortion
            identity_K = np.eye(3)
            success, rvec, tvec = cv2.solvePnP(
                obj_points, undistorted.reshape(-1, 2), 
                identity_K, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            # For pinhole cameras
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, 
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        return success, rvec, tvec
    
    def generate_board_image(self, pixels_per_meter: int = 2000, 
                            margin_pixels: int = 50) -> np.ndarray:
        """
        Generate a printable ChArUco board image.
        
        Args:
            pixels_per_meter: Resolution in pixels per meter
            margin_pixels: White margin around the board
            
        Returns:
            Board image as numpy array
        """
        # Calculate image size
        board_width_m = self.squares_x * self.square_length
        board_height_m = self.squares_y * self.square_length
        
        img_width = int(board_width_m * pixels_per_meter) + 2 * margin_pixels
        img_height = int(board_height_m * pixels_per_meter) + 2 * margin_pixels
        
        # Generate the board
        board_img = self.board.generateImage(
            (img_width - 2 * margin_pixels, img_height - 2 * margin_pixels)
        )
        
        # Add margin
        result = np.ones((img_height, img_width), dtype=np.uint8) * 255
        result[margin_pixels:margin_pixels + board_img.shape[0],
               margin_pixels:margin_pixels + board_img.shape[1]] = board_img
        
        return result
    
    def get_object_points(self) -> np.ndarray:
        """Get the 3D coordinates of all ChArUco corners."""
        return self.board.getChessboardCorners()
