##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/example_2_calibration/calibrate_stereo.py'
#   Example 2b: compute stereo calibration from captured chessboard
#   pairs and save it for the later examples.
#
#   Pipeline:
#     1. Find chessboard corners in every left/right pair.
#     2. Calibrate each camera individually (intrinsics K, distortion D).
#     3. cv2.stereoCalibrate -> rotation R and translation T between
#        the two cameras (T's magnitude is the BASELINE).
#     4. cv2.stereoRectify -> rectification transforms and the Q matrix
#        that converts disparity into metric 3D coordinates.
#     5. Save everything to stereo_calibration.npz.
#
#   Run:  python calibrate_stereo.py [--square-size MM]
#   The square size is the printed edge length of ONE chessboard
#   square in millimeters; it sets the absolute scale of the world.
##--------------------------------------------------------------------\

import argparse
from pathlib import Path

import cv2
import numpy as np

# Inner corner count (columns, rows). MUST match capture script.
#   (7, 7) = a REAL 8x8-square chessboard
#   (9, 6) = the classic printed OpenCV 10x7-square pattern
CHESSBOARD = (7, 7)
IMAGE_DIR = Path(__file__).resolve().parent / "calibration_images"
OUTPUT_FILE = Path(__file__).resolve().parent / "stereo_calibration.npz"


def find_corner_sets(square_size_mm):
    """Locate chessboard corners in every saved pair."""
    # Template of 3D board points in board coordinates (z = 0 plane)
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    object_points, left_points, right_points = [], [], []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    left_files = sorted(IMAGE_DIR.glob("left_*.png"))
    for left_file in left_files:
        right_file = IMAGE_DIR / left_file.name.replace("left", "right")
        if not right_file.exists():
            continue
        gray_l = cv2.imread(str(left_file), cv2.IMREAD_GRAYSCALE)
        gray_r = cv2.imread(str(right_file), cv2.IMREAD_GRAYSCALE)
        if gray_l is None or gray_r is None:
            continue
        image_size = gray_l.shape[::-1]

        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, None)
        if not (found_l and found_r):
            print(f"  skipping {left_file.name}: board not found in both")
            continue

        # Refine to sub-pixel accuracy
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        object_points.append(objp)
        left_points.append(corners_l)
        right_points.append(corners_r)
        print(f"  using {left_file.name} / {right_file.name}")

    return object_points, left_points, right_points, image_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--square-size", type=float, default=25.0,
                        help="chessboard square edge length in mm")
    args = parser.parse_args()

    print("Searching for chessboard corners...")
    objp, left_pts, right_pts, image_size = find_corner_sets(args.square_size)
    if len(objp) < 10:
        print(f"Only {len(objp)} usable pairs found - capture more "
              "(15-25 recommended) before calibrating.")
        return

    print(f"\nCalibrating individual cameras with {len(objp)} pairs...")
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(objp, left_pts, image_size, None, None)
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(objp, right_pts, image_size, None, None)
    print(f"  left RMS reprojection error:  {rms_l:.4f} px")
    print(f"  right RMS reprojection error: {rms_r:.4f} px")

    print("\nStereo calibration (relative pose between cameras)...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objp, left_pts, right_pts, K1, D1, K2, D2, image_size,
        criteria=criteria, flags=flags)
    baseline_mm = float(np.linalg.norm(T))
    print(f"  stereo RMS error: {rms_stereo:.4f} px")
    print(f"  baseline: {baseline_mm:.2f} mm "
          "(compare against a ruler measurement of the lens spacing!)")

    print("\nComputing rectification transforms...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0)

    np.savez(OUTPUT_FILE,
             image_size=image_size,
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             roi1=roi1, roi2=roi2,
             square_size_mm=args.square_size,
             rms_stereo=rms_stereo)
    print(f"\nSaved calibration to {OUTPUT_FILE}")
    print("Rule of thumb: stereo RMS under ~0.5 px is good; over ~1.0 px, "
          "recapture with sharper, more varied images.")


if __name__ == "__main__":
    main()