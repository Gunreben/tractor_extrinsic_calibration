#!/usr/bin/env python3
"""
Generate a printable ChArUco board for camera calibration.

This script generates a high-resolution ChArUco board image suitable for
printing on A1 paper (594mm x 841mm).

Usage:
    ros2 run camera_extrinsic_calibration generate_charuco_board.py [options]
    
    Or standalone:
    python3 generate_charuco_board.py --output charuco_board_A1.png
"""

import argparse
import cv2
import numpy as np
import os
import sys

# Add package to path for standalone execution
try:
    from camera_extrinsic_calibration.charuco_detector import CharucoDetector, ARUCO_DICT_MAP
    from camera_extrinsic_calibration.utils import load_charuco_config
except ImportError:
    # Standalone mode - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_extrinsic_calibration.charuco_detector import CharucoDetector, ARUCO_DICT_MAP
    from camera_extrinsic_calibration.utils import load_charuco_config


def generate_charuco_board(
    squares_x: int = 10,
    squares_y: int = 14,
    square_length_mm: float = 55.0,
    marker_length_mm: float = 42.0,
    aruco_dict: str = "DICT_6X6_250",
    dpi: int = 300,
    output_path: str = "charuco_board_A1.png",
    add_info: bool = True
):
    """
    Generate a ChArUco board image.
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length_mm: Square side length in mm
        marker_length_mm: ArUco marker side length in mm
        aruco_dict: ArUco dictionary name
        dpi: Output resolution in dots per inch
        output_path: Output file path
        add_info: Whether to add calibration info text to the image
    """
    # Convert mm to meters for internal use
    square_length = square_length_mm / 1000.0
    marker_length = marker_length_mm / 1000.0
    
    # Calculate pixels per meter based on DPI
    # 1 inch = 25.4 mm = 0.0254 m
    pixels_per_meter = dpi / 0.0254
    
    # Get ArUco dictionary
    if aruco_dict not in ARUCO_DICT_MAP:
        raise ValueError(f"Unknown ArUco dictionary: {aruco_dict}")
    
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[aruco_dict])
    
    # Create ChArUco board
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        dictionary
    )
    
    # Calculate board size in pixels
    board_width_m = squares_x * square_length
    board_height_m = squares_y * square_length
    
    board_width_px = int(board_width_m * pixels_per_meter)
    board_height_px = int(board_height_m * pixels_per_meter)
    
    # Add margin for printing
    margin_px = int(20 * dpi / 25.4)  # 20mm margin
    
    # Generate board image
    board_img = board.generateImage((board_width_px, board_height_px))
    
    # Create final image with margin
    if add_info:
        info_height = int(15 * dpi / 25.4)  # 15mm for info text
    else:
        info_height = 0
    
    final_width = board_width_px + 2 * margin_px
    final_height = board_height_px + 2 * margin_px + info_height
    
    final_img = np.ones((final_height, final_width), dtype=np.uint8) * 255
    
    # Place board in center
    y_offset = margin_px + info_height
    x_offset = margin_px
    final_img[y_offset:y_offset + board_height_px, 
              x_offset:x_offset + board_width_px] = board_img
    
    # Add calibration info text
    if add_info:
        # Convert to BGR for colored text
        final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
        
        info_text = (
            f"ChArUco Board | Squares: {squares_x}x{squares_y} | "
            f"Square: {square_length_mm}mm | Marker: {marker_length_mm}mm | "
            f"Dict: {aruco_dict}"
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = dpi / 150.0  # Scale font with DPI
        thickness = max(1, int(dpi / 150))
        
        # Get text size for centering
        (text_width, text_height), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
        
        text_x = (final_width - text_width) // 2
        text_y = margin_px // 2 + text_height // 2
        
        cv2.putText(final_img, info_text, (text_x, text_y), 
                   font, font_scale, (0, 0, 0), thickness)
        
        # Add corner markers for alignment verification
        corner_size = int(10 * dpi / 25.4)  # 10mm corner markers
        cv2.rectangle(final_img, (margin_px, y_offset), 
                     (margin_px + corner_size, y_offset + corner_size), (128, 128, 128), 2)
        cv2.rectangle(final_img, (final_width - margin_px - corner_size, y_offset),
                     (final_width - margin_px, y_offset + corner_size), (128, 128, 128), 2)
    
    # Save image
    cv2.imwrite(output_path, final_img)
    
    # Print info
    print(f"ChArUco board generated: {output_path}")
    print(f"  Board dimensions: {squares_x} x {squares_y} squares")
    print(f"  Square size: {square_length_mm} mm")
    print(f"  Marker size: {marker_length_mm} mm")
    print(f"  ArUco dictionary: {aruco_dict}")
    print(f"  Physical size: {board_width_m * 1000:.1f} x {board_height_m * 1000:.1f} mm")
    print(f"  Image resolution: {final_width} x {final_height} pixels @ {dpi} DPI")
    print()
    print("IMPORTANT: When printing, ensure:")
    print("  1. Print at 100% scale (no fit-to-page)")
    print("  2. Use matte paper to avoid reflections")
    print("  3. Mount on rigid, flat surface")
    print(f"  4. Verify square size with ruler: should be exactly {square_length_mm} mm")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate a printable ChArUco board for camera calibration'
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to charuco_board.yaml config file')
    parser.add_argument('--squares-x', type=int, default=10,
                       help='Number of squares in X direction (default: 10)')
    parser.add_argument('--squares-y', type=int, default=14,
                       help='Number of squares in Y direction (default: 14)')
    parser.add_argument('--square-length', type=float, default=55.0,
                       help='Square side length in mm (default: 55.0)')
    parser.add_argument('--marker-length', type=float, default=42.0,
                       help='Marker side length in mm (default: 42.0)')
    parser.add_argument('--aruco-dict', type=str, default='DICT_6X6_250',
                       choices=list(ARUCO_DICT_MAP.keys()),
                       help='ArUco dictionary (default: DICT_6X6_250)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution in DPI (default: 300)')
    parser.add_argument('--output', '-o', type=str, default='charuco_board_A1.png',
                       help='Output file path (default: charuco_board_A1.png)')
    parser.add_argument('--no-info', action='store_true',
                       help='Do not add info text to the image')
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        config = load_charuco_config(args.config)
        squares_x = config.get('squares_x', args.squares_x)
        squares_y = config.get('squares_y', args.squares_y)
        square_length = config.get('square_length', args.square_length / 1000.0) * 1000
        marker_length = config.get('marker_length', args.marker_length / 1000.0) * 1000
        aruco_dict = config.get('aruco_dictionary', args.aruco_dict)
    else:
        squares_x = args.squares_x
        squares_y = args.squares_y
        square_length = args.square_length
        marker_length = args.marker_length
        aruco_dict = args.aruco_dict
    
    generate_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length,
        marker_length_mm=marker_length,
        aruco_dict=aruco_dict,
        dpi=args.dpi,
        output_path=args.output,
        add_info=not args.no_info
    )


if __name__ == '__main__':
    main()
