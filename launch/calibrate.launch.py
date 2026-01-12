"""
Launch file for computing extrinsic calibration.

This is a helper launch file that sets up the environment for
running the compute_extrinsics.py script.

For most cases, running the script directly is recommended:
    python3 compute_extrinsics.py --help
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('camera_extrinsic_calibration')
    
    # Get tractor_multi_cam_publisher calibration directory (for intrinsics)
    try:
        tractor_pkg_share = get_package_share_directory('tractor_multi_cam_publisher')
        default_intrinsics_dir = os.path.join(tractor_pkg_share, 'calibration')
    except Exception:
        default_intrinsics_dir = '/tmp/calibration'
    
    # Default paths
    default_cameras_config = os.path.join(pkg_share, 'config', 'cameras.yaml')
    default_charuco_config = os.path.join(pkg_share, 'config', 'charuco_board.yaml')
    default_images_dir = '/tmp/calibration_images'
    default_output = '/tmp/extrinsics_calibrated.yaml'
    
    # Launch arguments
    cameras_config_arg = DeclareLaunchArgument(
        'cameras_config',
        default_value=default_cameras_config,
        description='Path to cameras.yaml'
    )
    
    charuco_config_arg = DeclareLaunchArgument(
        'charuco_config',
        default_value=default_charuco_config,
        description='Path to charuco_board.yaml'
    )
    
    images_dir_arg = DeclareLaunchArgument(
        'images_dir',
        default_value=default_images_dir,
        description='Directory containing calibration images'
    )
    
    intrinsics_dir_arg = DeclareLaunchArgument(
        'intrinsics_dir',
        default_value=default_intrinsics_dir,
        description='Directory containing intrinsics YAML files'
    )
    
    output_arg = DeclareLaunchArgument(
        'output',
        default_value=default_output,
        description='Output file path for extrinsics'
    )
    
    min_frames_arg = DeclareLaunchArgument(
        'min_frames',
        default_value='10',
        description='Minimum frames per camera pair'
    )
    
    # Execute the calibration script
    calibrate_cmd = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(pkg_share, 'lib', 'camera_extrinsic_calibration', 'compute_extrinsics.py'),
            '--cameras-config', LaunchConfiguration('cameras_config'),
            '--charuco-config', LaunchConfiguration('charuco_config'),
            '--images-dir', LaunchConfiguration('images_dir'),
            '--intrinsics-dir', LaunchConfiguration('intrinsics_dir'),
            '--output', LaunchConfiguration('output'),
            '--min-frames', LaunchConfiguration('min_frames'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        cameras_config_arg,
        charuco_config_arg,
        images_dir_arg,
        intrinsics_dir_arg,
        output_arg,
        min_frames_arg,
        calibrate_cmd,
    ])
