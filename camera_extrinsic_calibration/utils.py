"""
Utility functions for camera extrinsic calibration.
"""

import numpy as np
import yaml
import os
from typing import Dict, Any, Optional, Tuple


def load_camera_config(config_path: str) -> Dict[str, Any]:
    """Load camera configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_charuco_config(config_path: str) -> Dict[str, Any]:
    """Load ChArUco board configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_intrinsics(intrinsics_path: str) -> Dict[str, Any]:
    """
    Load camera intrinsics from YAML file.
    Supports the Main Street Autonomy calibration format.
    
    Returns dict with:
        - camera_matrix: 3x3 numpy array
        - dist_coeffs: distortion coefficients
        - distortion_model: string
        - image_size: (width, height)
        - is_fisheye: bool
    """
    with open(intrinsics_path, 'r') as f:
        content = f.read()
        # Handle the MSA format (starts with comments)
        lines = content.split('\n')
        yaml_lines = [l for l in lines if not l.strip().startswith('#')]
        data = yaml.safe_load('\n'.join(yaml_lines))
    
    # Extract camera matrix
    k_data = data['camera_matrix']['data']
    camera_matrix = np.array(k_data).reshape(3, 3)
    
    # Extract distortion coefficients
    d_data = data['distortion_coefficients']['data']
    dist_coeffs = np.array(d_data)
    
    # Get distortion model
    distortion_model = data.get('distortion_model', 'plumb_bob')
    
    # Determine if fisheye based on distortion model name
    # Common fisheye model names: equidistant, fisheye, kb4 (Kannala-Brandt)
    fisheye_models = ['equidistant', 'fisheye', 'kb4', 'kannala_brandt']
    is_fisheye = distortion_model.lower() in fisheye_models
    
    # OpenCV fisheye model requires exactly 4 distortion coefficients (k1, k2, k3, k4)
    # If we have more, truncate to 4 for fisheye
    if is_fisheye and len(dist_coeffs) > 4:
        print(f"  Note: Truncating {len(dist_coeffs)} distortion coefficients to 4 for fisheye model")
        dist_coeffs = dist_coeffs[:4]
    
    # Get image size
    image_size = (data['image_width'], data['image_height'])
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'distortion_model': distortion_model,
        'image_size': image_size,
        'camera_name': data.get('camera_name', 'unknown'),
        'is_fisheye': is_fisheye
    }


def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to quaternion [x, y, z, w].
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [x, y, z, w]
    """
    # Ensure R is a proper rotation matrix
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([x, y, z, w])


def matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    
    Args:
        q: Quaternion as [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q
    
    # Normalize quaternion
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def transform_to_matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Create 4x4 transformation matrix from translation and quaternion.
    
    Args:
        translation: [x, y, z] position
        quaternion: [x, y, z, w] rotation
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = matrix_from_quaternion(quaternion)
    T[:3, 3] = translation
    return T


def matrix_to_transform(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract translation and quaternion from 4x4 transformation matrix.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        Tuple of (translation [x,y,z], quaternion [x,y,z,w])
    """
    translation = T[:3, 3]
    quaternion = quaternion_from_matrix(T[:3, :3])
    return translation, quaternion


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 transformation matrix.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        Inverted 4x4 transformation matrix
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    
    return T_inv


def compose_transforms(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Compose two transformation matrices: T1 * T2
    
    Args:
        T1: First transformation (applied second)
        T2: Second transformation (applied first)
        
    Returns:
        Composed transformation matrix
    """
    return T1 @ T2


def save_extrinsics_yaml(extrinsics: Dict[str, Dict], output_path: str, 
                         reference_frame: str = "rear_mid"):
    """
    Save computed extrinsics to YAML file in the same format as existing extrinsics.yaml.
    
    Args:
        extrinsics: Dict mapping camera_name -> {translation, quaternion, parent, child}
        output_path: Path to save the YAML file
        reference_frame: Name of the reference frame
    """
    output = {
        '# Extrinsic calibration computed by camera_extrinsic_calibration package': None,
        f'# Reference frame: {reference_frame}': None,
        '# Format: [x_m, y_m, z_m, qx, qy, qz, qw]': None,
    }
    
    lines = [
        "# Extrinsic calibration computed by camera_extrinsic_calibration package",
        f"# Reference frame: {reference_frame}",
        "# Position xyz; Quaternions xyzw",
        "# [ x_m, y_m, z_m, qx, qy, qz, qw]",
    ]
    
    for cam_name, data in extrinsics.items():
        t = data['translation']
        q = data['quaternion']
        parent = data.get('parent', reference_frame)
        child = data.get('child', cam_name)
        
        lines.append(f"{cam_name}:")
        lines.append(f'  parent: "{parent}"')
        lines.append(f'  child: "{child}"')
        lines.append(f"  value: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}, {q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved extrinsics to {output_path}")


def save_extrinsics_urdf(extrinsics: Dict[str, Dict], output_path: str,
                         reference_frame: str = "rear_mid"):
    """
    Save computed extrinsics to URDF format for visualization.
    
    Args:
        extrinsics: Dict mapping camera_name -> {translation, quaternion}
        output_path: Path to save the URDF file
        reference_frame: Name of the reference frame
    """
    from scipy.spatial.transform import Rotation
    
    lines = [
        '<?xml version="1.0"?>',
        '<robot name="camera_extrinsics">',
        f'  <link name="{reference_frame}"/>',
    ]
    
    for cam_name, data in extrinsics.items():
        t = data['translation']
        q = data['quaternion']  # [x, y, z, w]
        
        # Convert quaternion to RPY for URDF
        r = Rotation.from_quat(q)  # scipy uses [x, y, z, w]
        rpy = r.as_euler('xyz')
        
        parent = data.get('parent', reference_frame)
        
        lines.extend([
            f'  <link name="{cam_name}"/>',
            f'  <joint name="{parent}_to_{cam_name}" type="fixed">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{cam_name}"/>',
            f'    <origin xyz="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>',
            '  </joint>',
        ])
    
    lines.append('</robot>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved URDF to {output_path}")
