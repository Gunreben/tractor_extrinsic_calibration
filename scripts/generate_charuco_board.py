#!/usr/bin/env python3
"""
Generate a printable ChArUco board for camera calibration.

This script generates a high-resolution ChArUco board suitable for
printing on standard paper sizes (A1, A2, A3, A4, etc.).

Supports PDF output (recommended) for correct physical dimensions,
or PNG output for image editing.

Usage:
    ros2 run camera_extrinsic_calibration generate_charuco_board.py [options]
    
    Or standalone:
    python3 generate_charuco_board.py --output charuco_board_A1.pdf
    python3 generate_charuco_board.py --output charuco_board_A1.png --paper-size A1
"""

import argparse
import cv2
import numpy as np
import os
import sys
from io import BytesIO

# Add package to path for standalone execution
try:
    from camera_extrinsic_calibration.charuco_detector import CharucoDetector, ARUCO_DICT_MAP
    from camera_extrinsic_calibration.utils import load_charuco_config
except ImportError:
    # Standalone mode - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera_extrinsic_calibration.charuco_detector import CharucoDetector, ARUCO_DICT_MAP
    from camera_extrinsic_calibration.utils import load_charuco_config


# Standard paper sizes in mm (width x height, portrait orientation)
PAPER_SIZES = {
    'A0': (841, 1189),
    'A1': (594, 841),
    'A2': (420, 594),
    'A3': (297, 420),
    'A4': (210, 297),
    'Letter': (216, 279),
    'Legal': (216, 356),
    'Tabloid': (279, 432),
}


def generate_charuco_board(
    squares_x: int = 10,
    squares_y: int = 14,
    square_length_mm: float = 55.0,
    marker_length_mm: float = 42.0,
    aruco_dict: str = "DICT_6X6_250",
    dpi: int = 300,
    output_path: str = "charuco_board_A1.pdf",
    add_info: bool = True,
    paper_size: str = None,
    margin_mm: float = 20.0
):
    """
    Generate a ChArUco board image or PDF.
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length_mm: Square side length in mm
        marker_length_mm: ArUco marker side length in mm
        aruco_dict: ArUco dictionary name
        dpi: Output resolution in dots per inch
        output_path: Output file path (.pdf or .png)
        add_info: Whether to add calibration info text
        paper_size: Paper size name (A1, A2, etc.) - required for PDF
        margin_mm: Margin around the board in mm
    """
    # Determine output format
    output_ext = os.path.splitext(output_path)[1].lower()
    is_pdf = output_ext == '.pdf'
    
    # Convert mm to meters for internal use
    square_length = square_length_mm / 1000.0
    marker_length = marker_length_mm / 1000.0
    
    # Calculate board physical size
    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm
    
    # Determine paper size
    if paper_size:
        if paper_size not in PAPER_SIZES:
            raise ValueError(f"Unknown paper size: {paper_size}. Available: {list(PAPER_SIZES.keys())}")
        paper_width_mm, paper_height_mm = PAPER_SIZES[paper_size]
    else:
        # Auto-detect paper size based on board dimensions
        paper_size = _find_best_paper_size(board_width_mm, board_height_mm, margin_mm)
        if paper_size:
            paper_width_mm, paper_height_mm = PAPER_SIZES[paper_size]
            print(f"Auto-selected paper size: {paper_size}")
        else:
            # Custom size - just fit the board with margins
            paper_width_mm = board_width_mm + 2 * margin_mm
            paper_height_mm = board_height_mm + 2 * margin_mm + (15 if add_info else 0)
            paper_size = "Custom"
    
    # Check if board fits on paper
    available_width = paper_width_mm - 2 * margin_mm
    available_height = paper_height_mm - 2 * margin_mm - (15 if add_info else 0)
    
    if board_width_mm > available_width or board_height_mm > available_height:
        print(f"\nWARNING: Board ({board_width_mm:.1f} x {board_height_mm:.1f} mm) is too large")
        print(f"         for {paper_size} paper ({paper_width_mm} x {paper_height_mm} mm)")
        print(f"         with {margin_mm}mm margins!")
        print(f"         Available space: {available_width:.1f} x {available_height:.1f} mm")
        print(f"\nOptions:")
        print(f"  1. Reduce square_length (currently {square_length_mm}mm)")
        print(f"  2. Reduce number of squares (currently {squares_x}x{squares_y})")
        print(f"  3. Use larger paper size")
        
        # Suggest fitting parameters
        max_square_x = available_width / squares_x
        max_square_y = available_height / squares_y
        suggested_square = min(max_square_x, max_square_y)
        print(f"\n  Suggested square_length for current grid: {suggested_square:.1f}mm")
        return None
    
    # Calculate pixels per meter based on DPI
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
    board_width_px = int(board_width_mm / 1000.0 * pixels_per_meter)
    board_height_px = int(board_height_mm / 1000.0 * pixels_per_meter)
    
    # Generate board image
    board_img = board.generateImage((board_width_px, board_height_px))
    
    if is_pdf:
        _save_as_pdf(
            board_img, output_path, 
            board_width_mm, board_height_mm,
            paper_width_mm, paper_height_mm,
            margin_mm, squares_x, squares_y,
            square_length_mm, marker_length_mm,
            aruco_dict, add_info
        )
    else:
        _save_as_image(
            board_img, output_path, dpi,
            board_width_px, board_height_px,
            margin_mm, squares_x, squares_y,
            square_length_mm, marker_length_mm,
            aruco_dict, add_info
        )
    
    # Print info
    print(f"\nChArUco board generated: {output_path}")
    print(f"  Board dimensions: {squares_x} x {squares_y} squares")
    print(f"  Square size: {square_length_mm} mm")
    print(f"  Marker size: {marker_length_mm} mm")
    print(f"  ArUco dictionary: {aruco_dict}")
    print(f"  Board physical size: {board_width_mm:.1f} x {board_height_mm:.1f} mm")
    print(f"  Paper size: {paper_size} ({paper_width_mm} x {paper_height_mm} mm)")
    
    if is_pdf:
        print(f"\nPDF output ensures correct physical dimensions.")
        print("When printing:")
        print("  1. Select 'Actual Size' or '100%' scale")
        print("  2. Do NOT use 'Fit to Page'")
    else:
        print(f"  Image resolution: {board_width_px} x {board_height_px} pixels @ {dpi} DPI")
        print("\nIMPORTANT: When printing PNG, ensure:")
        print("  1. Print at 100% scale (no fit-to-page)")
    
    print("  2. Use matte paper to avoid reflections")
    print("  3. Mount on rigid, flat surface")
    print(f"  4. Verify square size with ruler: should be exactly {square_length_mm} mm")
    
    return output_path


def _find_best_paper_size(board_width_mm: float, board_height_mm: float, margin_mm: float) -> str:
    """Find the smallest standard paper size that fits the board."""
    # Sort paper sizes by area (smallest first)
    sorted_sizes = sorted(PAPER_SIZES.items(), key=lambda x: x[1][0] * x[1][1])
    
    for name, (w, h) in sorted_sizes:
        # Check both orientations
        available_w = w - 2 * margin_mm
        available_h = h - 2 * margin_mm - 15  # Account for info text
        
        # Portrait
        if board_width_mm <= available_w and board_height_mm <= available_h:
            return name
        # Landscape (swap board dimensions conceptually)
        if board_height_mm <= available_w and board_width_mm <= available_h:
            return name
    
    return None


def _save_as_pdf(
    board_img, output_path: str,
    board_width_mm: float, board_height_mm: float,
    paper_width_mm: float, paper_height_mm: float,
    margin_mm: float, squares_x: int, squares_y: int,
    square_length_mm: float, marker_length_mm: float,
    aruco_dict: str, add_info: bool
):
    """Save the board as a PDF with exact physical dimensions."""
    try:
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from PIL import Image
    except ImportError:
        print("\nERROR: PDF output requires 'reportlab' and 'pillow' packages.")
        print("Install with: pip install reportlab pillow")
        sys.exit(1)
    
    # Create PDF with exact paper size
    c = canvas.Canvas(output_path, pagesize=(paper_width_mm * mm, paper_height_mm * mm))
    
    # Calculate centered position for the board
    info_height_mm = 12 if add_info else 0
    x_offset = (paper_width_mm - board_width_mm) / 2
    y_offset = (paper_height_mm - board_height_mm - info_height_mm) / 2
    
    # Convert OpenCV image to PIL Image
    pil_img = Image.fromarray(board_img)
    img_reader = ImageReader(pil_img)
    
    # Draw board image at exact physical size
    c.drawImage(
        img_reader,
        x_offset * mm,
        y_offset * mm,
        width=board_width_mm * mm,
        height=board_height_mm * mm
    )
    
    # Add info text at top
    if add_info:
        info_text = (
            f"ChArUco Board | Squares: {squares_x}x{squares_y} | "
            f"Square: {square_length_mm}mm | Marker: {marker_length_mm}mm | "
            f"Dict: {aruco_dict}"
        )
        
        c.setFont("Helvetica", 10)
        text_width = c.stringWidth(info_text, "Helvetica", 10)
        text_x = (paper_width_mm * mm - text_width) / 2
        text_y = paper_height_mm * mm - margin_mm * mm / 2
        
        c.drawString(text_x, text_y, info_text)
        
        # Add corner reference marks
        mark_size = 5 * mm
        board_left = x_offset * mm
        board_right = (x_offset + board_width_mm) * mm
        board_bottom = y_offset * mm
        board_top = (y_offset + board_height_mm) * mm
        
        c.setStrokeColorRGB(0.5, 0.5, 0.5)
        c.setLineWidth(0.5)
        
        # Top-left corner mark
        c.line(board_left - mark_size, board_top, board_left, board_top)
        c.line(board_left, board_top, board_left, board_top + mark_size)
        
        # Top-right corner mark
        c.line(board_right, board_top, board_right + mark_size, board_top)
        c.line(board_right, board_top, board_right, board_top + mark_size)
        
        # Bottom-left corner mark
        c.line(board_left - mark_size, board_bottom, board_left, board_bottom)
        c.line(board_left, board_bottom, board_left, board_bottom - mark_size)
        
        # Bottom-right corner mark
        c.line(board_right, board_bottom, board_right + mark_size, board_bottom)
        c.line(board_right, board_bottom, board_right, board_bottom - mark_size)
    
    c.save()


def _save_as_image(
    board_img, output_path: str, dpi: int,
    board_width_px: int, board_height_px: int,
    margin_mm: float, squares_x: int, squares_y: int,
    square_length_mm: float, marker_length_mm: float,
    aruco_dict: str, add_info: bool
):
    """Save the board as a PNG/image with margins and info text."""
    margin_px = int(margin_mm * dpi / 25.4)
    
    if add_info:
        info_height = int(15 * dpi / 25.4)
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
        final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
        
        info_text = (
            f"ChArUco Board | Squares: {squares_x}x{squares_y} | "
            f"Square: {square_length_mm}mm | Marker: {marker_length_mm}mm | "
            f"Dict: {aruco_dict}"
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = dpi / 150.0
        thickness = max(1, int(dpi / 150))
        
        (text_width, text_height), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
        
        text_x = (final_width - text_width) // 2
        text_y = margin_px // 2 + text_height // 2
        
        cv2.putText(final_img, info_text, (text_x, text_y), 
                   font, font_scale, (0, 0, 0), thickness)
        
        corner_size = int(10 * dpi / 25.4)
        cv2.rectangle(final_img, (margin_px, y_offset), 
                     (margin_px + corner_size, y_offset + corner_size), (128, 128, 128), 2)
        cv2.rectangle(final_img, (final_width - margin_px - corner_size, y_offset),
                     (final_width - margin_px, y_offset + corner_size), (128, 128, 128), 2)
    
    cv2.imwrite(output_path, final_img)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a printable ChArUco board for camera calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PDF for A1 paper (recommended):
  python generate_charuco_board.py -o charuco_board.pdf --paper-size A1

  # Generate with custom board parameters:
  python generate_charuco_board.py -o board.pdf --squares-x 8 --squares-y 11 --square-length 50 --paper-size A2

  # Generate PNG (less reliable for printing):
  python generate_charuco_board.py -o charuco_board.png --paper-size A1

  # Use config file:
  python generate_charuco_board.py --config config/charuco_board.yaml -o board.pdf
        """
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
    parser.add_argument('--output', '-o', type=str, default='charuco_board_A1.pdf',
                       help='Output file path (.pdf recommended, .png also supported)')
    parser.add_argument('--paper-size', type=str, default=None,
                       choices=list(PAPER_SIZES.keys()),
                       help='Paper size (auto-detected if not specified)')
    parser.add_argument('--margin', type=float, default=20.0,
                       help='Margin around board in mm (default: 20)')
    parser.add_argument('--no-info', action='store_true',
                       help='Do not add info text to the output')
    
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
        add_info=not args.no_info,
        paper_size=args.paper_size,
        margin_mm=args.margin
    )


if __name__ == '__main__':
    main()
