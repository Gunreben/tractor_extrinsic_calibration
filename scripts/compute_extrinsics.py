#!/usr/bin/env python3
"""
Compute extrinsic calibration from collected calibration images.

This script processes collected image pairs and computes the relative
poses between cameras using ChArUco board detections.

Usage:
    ros2 run camera_extrinsic_calibration compute_extrinsics.py \
        --cameras-config /path/to/cameras.yaml \
        --charuco-config /path/to/charuco_board.yaml \
        --images-dir /path/to/calibration_images \
        --intrinsics-dir /path/to/intrinsics \
        --output /path/to/output_extrinsics.yaml
"""

import argparse
import cv2
import numpy as np
import os
import sys
import glob
from typing import Dict, List, Tuple

# Add package to path for standalone execution
try:
    from camera_extrinsic_calibration.charuco_detector import CharucoDetector
    from camera_extrinsic_calibration.calibration_solver import ExtrinsicCalibrationSolver
    from camera_extrinsic_calibration.utils import (
        load_camera_config, load_charuco_config,
        save_extrinsics_yaml, save_extrinsics_urdf
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_extrinsic_calibration.charuco_detector import CharucoDetector
    from camera_extrinsic_calibration.calibration_solver import ExtrinsicCalibrationSolver
    from camera_extrinsic_calibration.utils import (
        load_camera_config, load_charuco_config,
        save_extrinsics_yaml, save_extrinsics_urdf
    )


def load_image_pairs(images_dir: str, cam1: str, cam2: str) -> List[Tuple[str, str]]:
    """
    Load image pairs for a camera pair.
    
    Returns list of (img1_path, img2_path) tuples.
    """
    pair_dir = os.path.join(images_dir, f"{cam1}_{cam2}")
    
    if not os.path.exists(pair_dir):
        # Try reverse order
        pair_dir = os.path.join(images_dir, f"{cam2}_{cam1}")
        if not os.path.exists(pair_dir):
            return []
    
    # Find all image files
    all_files = sorted(os.listdir(pair_dir))
    
    # Group by timestamp
    pairs = []
    timestamps = set()
    
    for f in all_files:
        if f.endswith('.png') or f.endswith('.jpg'):
            # Extract timestamp (format: YYYYMMDD_HHMMSS_camname.png)
            parts = f.rsplit('_', 1)
            if len(parts) == 2:
                timestamp = parts[0]
                timestamps.add(timestamp)
    
    for ts in sorted(timestamps):
        img1_path = None
        img2_path = None
        
        for f in all_files:
            if f.startswith(ts):
                if cam1 in f:
                    img1_path = os.path.join(pair_dir, f)
                elif cam2 in f:
                    img2_path = os.path.join(pair_dir, f)
        
        if img1_path and img2_path:
            pairs.append((img1_path, img2_path))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Compute extrinsic calibration from collected images'
    )
    
    parser.add_argument('--cameras-config', type=str, required=True,
                       help='Path to cameras.yaml config file')
    parser.add_argument('--charuco-config', type=str, required=True,
                       help='Path to charuco_board.yaml config file')
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing collected calibration images')
    parser.add_argument('--intrinsics-dir', type=str, required=True,
                       help='Directory containing intrinsics YAML files')
    parser.add_argument('--output', '-o', type=str, default='extrinsics_calibrated.yaml',
                       help='Output file path for extrinsics (default: extrinsics_calibrated.yaml)')
    parser.add_argument('--output-urdf', type=str, default=None,
                       help='Output URDF file path (optional)')
    parser.add_argument('--min-frames', type=int, default=10,
                       help='Minimum number of frame pairs per camera pair (default: 10)')
    parser.add_argument('--min-corners', type=int, default=8,
                       help='Minimum detected corners per frame (default: 8)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of detections')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    cameras_config = load_camera_config(args.cameras_config)
    charuco_config = load_charuco_config(args.charuco_config)
    
    # Initialize detector and solver
    print("Initializing calibration system...")
    detector = CharucoDetector(charuco_config)
    solver = ExtrinsicCalibrationSolver(detector, cameras_config, args.intrinsics_dir)
    
    # Get calibration pairs
    calibration_pairs = cameras_config.get('calibration_pairs', [])
    
    if not calibration_pairs:
        print("ERROR: No calibration pairs defined in config")
        sys.exit(1)
    
    print(f"\nCalibration pairs: {calibration_pairs}")
    print(f"Reference camera: {cameras_config.get('reference_camera', 'unknown')}")
    print()
    
    # Process each camera pair
    total_frames = 0
    
    for cam1, cam2 in calibration_pairs:
        print(f"\n{'='*60}")
        print(f"Processing pair: {cam1} <-> {cam2}")
        print('='*60)
        
        # Load image pairs
        image_pairs = load_image_pairs(args.images_dir, cam1, cam2)
        
        if not image_pairs:
            print(f"  WARNING: No image pairs found for {cam1}-{cam2}")
            continue
        
        print(f"  Found {len(image_pairs)} image pairs")
        
        # Process each pair
        successful = 0
        for img1_path, img2_path in image_pairs:
            # Load images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                if args.verbose:
                    print(f"  Failed to load: {img1_path} or {img2_path}")
                continue
            
            # Process frame pair
            success, data = solver.process_frame_pair(
                cam1, cam2, img1, img2, 
                min_corners=args.min_corners
            )
            
            if success:
                successful += 1
                if args.verbose:
                    print(f"  ✓ Processed: {os.path.basename(img1_path)} "
                          f"({data['common_corners']} common corners)")
                
                if args.visualize and data:
                    # Show detections
                    vis1 = data['det1'].image_with_detections
                    vis2 = data['det2'].image_with_detections
                    
                    if vis1 is not None and vis2 is not None:
                        h1, w1 = vis1.shape[:2]
                        h2, w2 = vis2.shape[:2]
                        
                        # Resize for display
                        scale = 600 / max(h1, h2)
                        vis1 = cv2.resize(vis1, None, fx=scale, fy=scale)
                        vis2 = cv2.resize(vis2, None, fx=scale, fy=scale)
                        
                        combined = np.hstack([vis1, vis2])
                        cv2.imshow('Detections', combined)
                        key = cv2.waitKey(100)
                        if key == ord('q'):
                            args.visualize = False
            else:
                if args.verbose:
                    reason = data.get('reason', 'unknown') if data else 'unknown'
                    print(f"  ✗ Failed: {os.path.basename(img1_path)} ({reason})")
        
        print(f"  Successfully processed: {successful}/{len(image_pairs)} pairs")
        total_frames += successful
    
    if args.visualize:
        cv2.destroyAllWindows()
    
    # Print statistics
    print("\n" + "="*60)
    print("CALIBRATION DATA SUMMARY")
    print("="*60)
    stats = solver.get_statistics()
    for pair_name, pair_stats in stats.items():
        print(f"  {pair_name}: {pair_stats['num_frames']} frames, "
              f"avg {pair_stats['avg_common_corners']:.1f} common corners")
    
    print(f"\nTotal processed frames: {total_frames}")
    
    # Compute calibration
    print("\n" + "="*60)
    print("COMPUTING EXTRINSIC CALIBRATION")
    print("="*60)
    
    min_frames_calibration = min(args.min_frames, max(5, total_frames // len(calibration_pairs) // 2))
    
    extrinsics = solver.compute_full_calibration(min_frames=min_frames_calibration)
    
    if not extrinsics:
        print("\nERROR: Calibration failed - no valid extrinsics computed")
        print("Possible causes:")
        print("  - Insufficient image pairs")
        print("  - Poor ChArUco detection")
        print("  - Missing camera intrinsics")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    
    reference = cameras_config.get('reference_camera', 'rear_mid')
    
    for cam_name, data in extrinsics.items():
        t = data['translation']
        q = data['quaternion']
        print(f"\n{cam_name} (relative to {reference}):")
        print(f"  Translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}] m")
        print(f"  Quaternion:  [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
        print(f"  Distance:    {np.linalg.norm(t):.4f} m")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    save_extrinsics_yaml(extrinsics, args.output, reference)
    
    if args.output_urdf:
        save_extrinsics_urdf(extrinsics, args.output_urdf, reference)
    
    print("\n✓ Calibration complete!")
    print(f"  Extrinsics saved to: {args.output}")
    if args.output_urdf:
        print(f"  URDF saved to: {args.output_urdf}")


if __name__ == '__main__':
    main()
