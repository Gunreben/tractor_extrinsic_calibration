"""
Launch file for collecting calibration images.

This launch file starts the image collector node that subscribes to
camera topics and captures synchronized images when the ChArUco board
is visible in overlapping camera pairs.

Usage:
    ros2 launch camera_extrinsic_calibration collect_images.launch.py
    
    Or with custom paths:
    ros2 launch camera_extrinsic_calibration collect_images.launch.py \
        output_dir:=/path/to/save/images
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('camera_extrinsic_calibration')
    
    # Default config paths
    default_cameras_config = os.path.join(pkg_share, 'config', 'cameras.yaml')
    default_charuco_config = os.path.join(pkg_share, 'config', 'charuco_board.yaml')
    default_output_dir = '/tmp/calibration_images'
    
    # Declare launch arguments
    cameras_config_arg = DeclareLaunchArgument(
        'cameras_config',
        default_value=default_cameras_config,
        description='Path to cameras.yaml configuration file'
    )
    
    charuco_config_arg = DeclareLaunchArgument(
        'charuco_config',
        default_value=default_charuco_config,
        description='Path to charuco_board.yaml configuration file'
    )
    
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value=default_output_dir,
        description='Directory to save calibration images'
    )
    
    sync_tolerance_arg = DeclareLaunchArgument(
        'sync_tolerance_ms',
        default_value='100.0',
        description='Time synchronization tolerance in milliseconds'
    )
    
    min_corners_arg = DeclareLaunchArgument(
        'min_corners',
        default_value='8',
        description='Minimum number of detected ChArUco corners'
    )
    
    auto_capture_arg = DeclareLaunchArgument(
        'auto_capture',
        default_value='false',
        description='Enable automatic capture at regular intervals'
    )
    
    auto_capture_interval_arg = DeclareLaunchArgument(
        'auto_capture_interval',
        default_value='2.0',
        description='Auto capture interval in seconds'
    )
    
    # Image collector node
    image_collector_node = Node(
        package='camera_extrinsic_calibration',
        executable='collect_calibration_images.py',
        name='image_collector',
        output='screen',
        parameters=[{
            'cameras_config': LaunchConfiguration('cameras_config'),
            'charuco_config': LaunchConfiguration('charuco_config'),
            'output_dir': LaunchConfiguration('output_dir'),
            'sync_tolerance_ms': LaunchConfiguration('sync_tolerance_ms'),
            'min_corners': LaunchConfiguration('min_corners'),
            'auto_capture': LaunchConfiguration('auto_capture'),
            'auto_capture_interval': LaunchConfiguration('auto_capture_interval'),
        }]
    )
    
    return LaunchDescription([
        cameras_config_arg,
        charuco_config_arg,
        output_dir_arg,
        sync_tolerance_arg,
        min_corners_arg,
        auto_capture_arg,
        auto_capture_interval_arg,
        image_collector_node,
    ])
