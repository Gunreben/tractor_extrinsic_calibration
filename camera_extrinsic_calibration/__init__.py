# camera_extrinsic_calibration package
"""
Multi-camera extrinsic calibration using ChArUco boards.

This package provides tools for calibrating the extrinsic parameters
(relative poses) between multiple cameras with overlapping fields of view.
"""

from .charuco_detector import CharucoDetector
from .calibration_solver import ExtrinsicCalibrationSolver
from .utils import load_camera_config, load_intrinsics, quaternion_from_matrix

__all__ = [
    'CharucoDetector',
    'ExtrinsicCalibrationSolver', 
    'load_camera_config',
    'load_intrinsics',
    'quaternion_from_matrix',
]
