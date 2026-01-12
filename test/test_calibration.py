#!/usr/bin/env python3
"""
Unit tests for camera extrinsic calibration package.
"""

import unittest
import numpy as np
import os
import sys

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_extrinsic_calibration.utils import (
    quaternion_from_matrix,
    matrix_from_quaternion,
    transform_to_matrix,
    matrix_to_transform,
    invert_transform,
    compose_transforms,
)


class TestTransformUtils(unittest.TestCase):
    """Test transformation utility functions."""
    
    def test_quaternion_roundtrip(self):
        """Test quaternion to matrix and back."""
        # Identity rotation
        R_identity = np.eye(3)
        q = quaternion_from_matrix(R_identity)
        R_back = matrix_from_quaternion(q)
        np.testing.assert_array_almost_equal(R_identity, R_back, decimal=6)
        
        # Random rotation
        from scipy.spatial.transform import Rotation
        R_random = Rotation.random().as_matrix()
        q = quaternion_from_matrix(R_random)
        R_back = matrix_from_quaternion(q)
        np.testing.assert_array_almost_equal(R_random, R_back, decimal=6)
    
    def test_transform_roundtrip(self):
        """Test transform to matrix and back."""
        translation = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([0.0, 0.0, 0.707, 0.707])  # 90 deg around Z
        quaternion = quaternion / np.linalg.norm(quaternion)
        
        T = transform_to_matrix(translation, quaternion)
        t_back, q_back = matrix_to_transform(T)
        
        np.testing.assert_array_almost_equal(translation, t_back, decimal=6)
        # Quaternions can have opposite sign and be equivalent
        if np.dot(quaternion, q_back) < 0:
            q_back = -q_back
        np.testing.assert_array_almost_equal(quaternion, q_back, decimal=6)
    
    def test_invert_transform(self):
        """Test transform inversion."""
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        from scipy.spatial.transform import Rotation
        T[:3, :3] = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        
        T_inv = invert_transform(T)
        T_identity = compose_transforms(T, T_inv)
        
        np.testing.assert_array_almost_equal(T_identity, np.eye(4), decimal=6)
    
    def test_compose_transforms(self):
        """Test transform composition."""
        # Two translations
        T1 = np.eye(4)
        T1[:3, 3] = [1.0, 0.0, 0.0]
        
        T2 = np.eye(4)
        T2[:3, 3] = [0.0, 1.0, 0.0]
        
        T_composed = compose_transforms(T1, T2)
        
        # T1 * T2 should translate by (1, 1, 0)
        expected_translation = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(
            T_composed[:3, 3], expected_translation, decimal=6
        )


class TestCharucoDetector(unittest.TestCase):
    """Test ChArUco detector."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized with config."""
        from camera_extrinsic_calibration.charuco_detector import CharucoDetector
        
        config = {
            'squares_x': 6,
            'squares_y': 8,
            'square_length': 0.04,
            'marker_length': 0.03,
            'aruco_dictionary': 'DICT_6X6_250'
        }
        
        detector = CharucoDetector(config)
        self.assertEqual(detector.squares_x, 6)
        self.assertEqual(detector.squares_y, 8)
    
    def test_board_generation(self):
        """Test board image generation."""
        from camera_extrinsic_calibration.charuco_detector import CharucoDetector
        
        config = {
            'squares_x': 6,
            'squares_y': 8,
            'square_length': 0.04,
            'marker_length': 0.03,
            'aruco_dictionary': 'DICT_6X6_250'
        }
        
        detector = CharucoDetector(config)
        board_img = detector.generate_board_image(pixels_per_meter=1000)
        
        self.assertIsNotNone(board_img)
        self.assertEqual(len(board_img.shape), 2)  # Grayscale
        self.assertGreater(board_img.shape[0], 100)
        self.assertGreater(board_img.shape[1], 100)


if __name__ == '__main__':
    unittest.main()
