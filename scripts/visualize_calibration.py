#!/usr/bin/env python3
"""
Visualize extrinsic calibration results.

This script creates a 3D visualization of the camera poses
and can project points between cameras to verify calibration quality.

Usage:
    ros2 run camera_extrinsic_calibration visualize_calibration.py \
        --extrinsics /path/to/extrinsics.yaml
"""

import argparse
import numpy as np
import yaml
import sys
import os

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_extrinsics(filepath: str) -> dict:
    """Load extrinsics from YAML file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse YAML, skipping comment lines
    lines = content.split('\n')
    yaml_lines = []
    for line in lines:
        if not line.strip().startswith('#'):
            yaml_lines.append(line)
    
    data = yaml.safe_load('\n'.join(yaml_lines))
    return data if data else {}


def plot_camera_frame(ax, T, name, scale=0.1, color='b'):
    """Plot a camera coordinate frame."""
    # Origin
    origin = T[:3, 3]
    
    # Axes (columns of rotation matrix)
    R = T[:3, :3]
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale
    
    # Plot axes
    ax.quiver(*origin, *x_axis, color='r', arrow_length_ratio=0.1)
    ax.quiver(*origin, *y_axis, color='g', arrow_length_ratio=0.1)
    ax.quiver(*origin, *z_axis, color='b', arrow_length_ratio=0.1)
    
    # Plot camera position
    ax.scatter(*origin, s=50, c=color, marker='o')
    ax.text(origin[0], origin[1], origin[2] + scale * 0.5, name, fontsize=8)


def visualize_extrinsics(extrinsics: dict, reference_frame: str = "rear_mid"):
    """Create 3D visualization of camera poses."""
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different cameras
    colors = plt.cm.tab10(np.linspace(0, 1, len(extrinsics) + 1))
    
    # Plot reference frame at origin
    T_ref = np.eye(4)
    plot_camera_frame(ax, T_ref, reference_frame, scale=0.15, color=colors[0])
    
    # Plot all other cameras
    for idx, (cam_name, data) in enumerate(extrinsics.items()):
        value = data.get('value', [])
        if len(value) != 7:
            print(f"Warning: Invalid extrinsics for {cam_name}")
            continue
        
        x, y, z, qx, qy, qz, qw = value
        
        # Build transformation matrix
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        
        # Quaternion to rotation matrix
        q = np.array([qx, qy, qz, qw])
        q = q / np.linalg.norm(q)
        
        # Quaternion to rotation matrix (x, y, z, w convention)
        x, y, z, w = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        T[:3, :3] = R
        
        plot_camera_frame(ax, T, cam_name, scale=0.15, color=colors[idx + 1])
    
    # Set axis properties
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Extrinsic Calibration Visualization')
    
    # Equal aspect ratio
    max_range = 2.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add legend
    ax.legend(['X axis', 'Y axis', 'Z axis'], loc='upper right')
    
    plt.tight_layout()
    plt.show()


def print_extrinsics_summary(extrinsics: dict, reference_frame: str = "rear_mid"):
    """Print a summary of the extrinsics."""
    print("\n" + "="*70)
    print("EXTRINSIC CALIBRATION SUMMARY")
    print("="*70)
    print(f"Reference frame: {reference_frame}")
    print()
    
    print(f"{'Camera':<25} {'Distance (m)':<12} {'Position (x, y, z)'}")
    print("-"*70)
    
    for cam_name, data in extrinsics.items():
        value = data.get('value', [])
        if len(value) != 7:
            continue
        
        x, y, z = value[0], value[1], value[2]
        distance = np.sqrt(x**2 + y**2 + z**2)
        
        print(f"{cam_name:<25} {distance:<12.4f} ({x:.4f}, {y:.4f}, {z:.4f})")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize extrinsic calibration results'
    )
    
    parser.add_argument('--extrinsics', '-e', type=str, required=True,
                       help='Path to extrinsics YAML file')
    parser.add_argument('--reference', '-r', type=str, default='rear_mid',
                       help='Reference frame name (default: rear_mid)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip 3D visualization')
    
    args = parser.parse_args()
    
    # Load extrinsics
    if not os.path.exists(args.extrinsics):
        print(f"ERROR: File not found: {args.extrinsics}")
        sys.exit(1)
    
    extrinsics = load_extrinsics(args.extrinsics)
    
    if not extrinsics:
        print("ERROR: No extrinsics data found in file")
        sys.exit(1)
    
    # Print summary
    print_extrinsics_summary(extrinsics, args.reference)
    
    # Visualize
    if not args.no_plot:
        visualize_extrinsics(extrinsics, args.reference)


if __name__ == '__main__':
    main()
