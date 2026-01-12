"""
Extrinsic calibration solver for multi-camera systems.
Computes relative poses between cameras using ChArUco board detections.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from .charuco_detector import CharucoDetector, DetectionResult
from .utils import (
    quaternion_from_matrix, matrix_from_quaternion,
    transform_to_matrix, matrix_to_transform, invert_transform, compose_transforms
)


@dataclass
class PairwiseCalibrationResult:
    """Result of pairwise extrinsic calibration between two cameras."""
    success: bool
    source_camera: str
    target_camera: str
    rotation_matrix: Optional[np.ndarray] = None  # 3x3 rotation from source to target
    translation: Optional[np.ndarray] = None       # Translation from source to target
    transform_matrix: Optional[np.ndarray] = None  # 4x4 transform matrix
    reprojection_error: float = float('inf')
    num_frames_used: int = 0


class ExtrinsicCalibrationSolver:
    """
    Solver for multi-camera extrinsic calibration.
    
    Uses ChArUco board detections to compute relative poses between cameras.
    """
    
    def __init__(self, charuco_detector: CharucoDetector, 
                 cameras_config: Dict[str, Any],
                 intrinsics_dir: str):
        """
        Initialize the calibration solver.
        
        Args:
            charuco_detector: Configured ChArUco detector
            cameras_config: Camera configuration dictionary
            intrinsics_dir: Directory containing intrinsics YAML files
        """
        self.detector = charuco_detector
        self.cameras_config = cameras_config
        self.intrinsics_dir = intrinsics_dir
        
        # Cache for loaded intrinsics
        self._intrinsics_cache: Dict[str, Dict] = {}
        
        # Storage for collected calibration data
        # Structure: {(cam1, cam2): [list of (detection1, detection2, pose1, pose2) tuples]}
        self.calibration_data: Dict[Tuple[str, str], List] = {}
    
    def get_intrinsics(self, camera_name: str) -> Dict[str, Any]:
        """Load and cache intrinsics for a camera."""
        if camera_name in self._intrinsics_cache:
            return self._intrinsics_cache[camera_name]
        
        from .utils import load_intrinsics
        import os
        
        cam_config = self.cameras_config['cameras'][camera_name]
        intrinsics_file = cam_config.get('intrinsics_file')
        
        if intrinsics_file is None:
            raise ValueError(f"No intrinsics file specified for camera {camera_name}")
        
        intrinsics_path = os.path.join(self.intrinsics_dir, intrinsics_file)
        intrinsics = load_intrinsics(intrinsics_path)
        
        # Add distortion model info from config
        intrinsics['is_fisheye'] = cam_config.get('distortion_model', 'pinhole') == 'fisheye'
        
        self._intrinsics_cache[camera_name] = intrinsics
        return intrinsics
    
    def process_frame_pair(self, cam1_name: str, cam2_name: str,
                           image1: np.ndarray, image2: np.ndarray,
                           min_corners: int = 6) -> Tuple[bool, Optional[Dict]]:
        """
        Process a frame pair from two cameras.
        
        Args:
            cam1_name: Name of first camera
            cam2_name: Name of second camera
            image1: Image from first camera
            image2: Image from second camera
            min_corners: Minimum number of corners required for valid detection
            
        Returns:
            Tuple of (success, data_dict with detections and poses)
        """
        # Get intrinsics
        try:
            intr1 = self.get_intrinsics(cam1_name)
            intr2 = self.get_intrinsics(cam2_name)
        except Exception as e:
            print(f"Error loading intrinsics: {e}")
            return False, None
        
        # Detect ChArUco in both images
        det1 = self.detector.detect(
            image1, 
            intr1['camera_matrix'], 
            intr1['dist_coeffs'],
            intr1['is_fisheye'],
            draw_detections=True
        )
        
        det2 = self.detector.detect(
            image2,
            intr2['camera_matrix'],
            intr2['dist_coeffs'],
            intr2['is_fisheye'],
            draw_detections=True
        )
        
        # Check if both detections have enough corners
        if not det1.success or det1.num_corners < min_corners:
            return False, {'det1': det1, 'det2': det2, 'reason': 'cam1_detection_failed'}
        
        if not det2.success or det2.num_corners < min_corners:
            return False, {'det1': det1, 'det2': det2, 'reason': 'cam2_detection_failed'}
        
        # Find common corners (corners seen by both cameras)
        common_ids = np.intersect1d(det1.charuco_ids, det2.charuco_ids)
        
        if len(common_ids) < min_corners:
            return False, {'det1': det1, 'det2': det2, 'reason': 'insufficient_common_corners'}
        
        # Estimate pose for both cameras
        success1, rvec1, tvec1 = self.detector.estimate_pose(
            det1.charuco_corners, det1.charuco_ids,
            intr1['camera_matrix'], intr1['dist_coeffs'],
            intr1['is_fisheye']
        )
        
        success2, rvec2, tvec2 = self.detector.estimate_pose(
            det2.charuco_corners, det2.charuco_ids,
            intr2['camera_matrix'], intr2['dist_coeffs'],
            intr2['is_fisheye']
        )
        
        if not success1 or not success2:
            return False, {'det1': det1, 'det2': det2, 'reason': 'pose_estimation_failed'}
        
        # Store the data
        data = {
            'det1': det1,
            'det2': det2,
            'rvec1': rvec1,
            'tvec1': tvec1,
            'rvec2': rvec2,
            'tvec2': tvec2,
            'common_corners': len(common_ids)
        }
        
        pair_key = (cam1_name, cam2_name)
        if pair_key not in self.calibration_data:
            self.calibration_data[pair_key] = []
        self.calibration_data[pair_key].append(data)
        
        return True, data
    
    def compute_pairwise_extrinsics(self, cam1_name: str, cam2_name: str,
                                    min_frames: int = 10) -> PairwiseCalibrationResult:
        """
        Compute extrinsic calibration between two cameras.
        
        The transform represents T_cam2_cam1: the pose of cam1 in cam2's frame.
        
        Args:
            cam1_name: Name of first (source) camera
            cam2_name: Name of second (target) camera
            min_frames: Minimum number of frame pairs required
            
        Returns:
            PairwiseCalibrationResult
        """
        pair_key = (cam1_name, cam2_name)
        
        if pair_key not in self.calibration_data:
            return PairwiseCalibrationResult(
                success=False,
                source_camera=cam1_name,
                target_camera=cam2_name
            )
        
        data_list = self.calibration_data[pair_key]
        
        if len(data_list) < min_frames:
            print(f"Insufficient frames for {cam1_name} -> {cam2_name}: "
                  f"{len(data_list)} < {min_frames}")
            return PairwiseCalibrationResult(
                success=False,
                source_camera=cam1_name,
                target_camera=cam2_name,
                num_frames_used=len(data_list)
            )
        
        # Compute relative transform for each frame pair
        transforms = []
        
        for data in data_list:
            # T_board_cam1: transform from cam1 to board
            R1, _ = cv2.Rodrigues(data['rvec1'])
            t1 = data['tvec1'].flatten()
            T_board_cam1 = np.eye(4)
            T_board_cam1[:3, :3] = R1
            T_board_cam1[:3, 3] = t1
            
            # T_board_cam2: transform from cam2 to board
            R2, _ = cv2.Rodrigues(data['rvec2'])
            t2 = data['tvec2'].flatten()
            T_board_cam2 = np.eye(4)
            T_board_cam2[:3, :3] = R2
            T_board_cam2[:3, 3] = t2
            
            # T_cam2_cam1 = T_cam2_board * T_board_cam1
            #             = inv(T_board_cam2) * T_board_cam1
            T_cam1_board = invert_transform(T_board_cam1)
            T_cam2_cam1 = compose_transforms(T_board_cam2, T_cam1_board)
            
            transforms.append(T_cam2_cam1)
        
        # Average the transforms (using quaternion averaging for rotation)
        avg_transform = self._average_transforms(transforms)
        
        # Compute reprojection error (simplified - using translation variance)
        translations = np.array([T[:3, 3] for T in transforms])
        translation_std = np.std(translations, axis=0)
        reprojection_error = np.linalg.norm(translation_std)
        
        return PairwiseCalibrationResult(
            success=True,
            source_camera=cam1_name,
            target_camera=cam2_name,
            rotation_matrix=avg_transform[:3, :3],
            translation=avg_transform[:3, 3],
            transform_matrix=avg_transform,
            reprojection_error=reprojection_error,
            num_frames_used=len(data_list)
        )
    
    def _average_transforms(self, transforms: List[np.ndarray]) -> np.ndarray:
        """
        Average multiple transformation matrices.
        Uses quaternion averaging for rotations and arithmetic mean for translations.
        """
        # Extract quaternions and translations
        quaternions = []
        translations = []
        
        for T in transforms:
            q = quaternion_from_matrix(T[:3, :3])
            quaternions.append(q)
            translations.append(T[:3, 3])
        
        quaternions = np.array(quaternions)
        translations = np.array(translations)
        
        # Average translation
        avg_translation = np.mean(translations, axis=0)
        
        # Average quaternion (ensuring consistent sign)
        # Make sure all quaternions have same hemisphere
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] = -quaternions[i]
        
        avg_quaternion = np.mean(quaternions, axis=0)
        avg_quaternion = avg_quaternion / np.linalg.norm(avg_quaternion)
        
        # Reconstruct transform
        avg_transform = np.eye(4)
        avg_transform[:3, :3] = matrix_from_quaternion(avg_quaternion)
        avg_transform[:3, 3] = avg_translation
        
        return avg_transform
    
    def compute_full_calibration(self, min_frames: int = 10) -> Dict[str, Dict]:
        """
        Compute full extrinsic calibration following the calibration chain.
        
        Returns transforms from each camera to the reference camera.
        
        Args:
            min_frames: Minimum frames per camera pair
            
        Returns:
            Dict mapping camera_name -> {translation, quaternion, parent, child}
        """
        reference = self.cameras_config['reference_camera']
        pairs = self.cameras_config['calibration_pairs']
        
        # Build adjacency for the calibration graph
        adjacency = {}
        for src, tgt in pairs:
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(tgt)
            # Also add reverse for traversal
            if tgt not in adjacency:
                adjacency[tgt] = []
            adjacency[tgt].append(src)
        
        # Compute pairwise calibrations
        pairwise_results = {}
        for src, tgt in pairs:
            result = self.compute_pairwise_extrinsics(src, tgt, min_frames)
            if result.success:
                pairwise_results[(src, tgt)] = result
                print(f"✓ {src} -> {tgt}: error={result.reprojection_error:.4f}m, "
                      f"frames={result.num_frames_used}")
            else:
                print(f"✗ {src} -> {tgt}: FAILED (frames={result.num_frames_used})")
        
        # Propagate transforms from reference using BFS
        transforms_to_ref = {reference: np.eye(4)}
        parent_map = {reference: None}
        visited = {reference}
        queue = [reference]
        
        while queue:
            current = queue.pop(0)
            
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                
                # Find the pairwise transform
                if (current, neighbor) in pairwise_results:
                    # T_neighbor_current
                    T_rel = pairwise_results[(current, neighbor)].transform_matrix
                elif (neighbor, current) in pairwise_results:
                    # T_current_neighbor -> need to invert
                    T_rel = invert_transform(
                        pairwise_results[(neighbor, current)].transform_matrix
                    )
                else:
                    print(f"Warning: No calibration between {current} and {neighbor}")
                    continue
                
                # T_neighbor_ref = T_neighbor_current * T_current_ref
                T_current_ref = transforms_to_ref[current]
                T_neighbor_ref = compose_transforms(T_rel, T_current_ref)
                
                transforms_to_ref[neighbor] = T_neighbor_ref
                parent_map[neighbor] = current
                visited.add(neighbor)
                queue.append(neighbor)
        
        # Convert to output format
        extrinsics = {}
        for cam_name, T in transforms_to_ref.items():
            if cam_name == reference:
                continue
            
            translation, quaternion = matrix_to_transform(T)
            
            extrinsics[cam_name] = {
                'translation': translation,
                'quaternion': quaternion,
                'parent': reference,
                'child': cam_name,
                'transform_matrix': T
            }
        
        return extrinsics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected calibration data."""
        stats = {}
        for pair_key, data_list in self.calibration_data.items():
            cam1, cam2 = pair_key
            stats[f"{cam1}->{cam2}"] = {
                'num_frames': len(data_list),
                'avg_common_corners': np.mean([d['common_corners'] for d in data_list]) if data_list else 0
            }
        return stats
    
    def clear_data(self):
        """Clear all collected calibration data."""
        self.calibration_data.clear()
