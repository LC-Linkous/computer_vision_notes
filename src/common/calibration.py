##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/common/calibration.py'
#   Loads the stereo calibration produced by Example 2 and precomputes
#   rectification maps. Used by Examples 3, 4, and 5.
##--------------------------------------------------------------------\

from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB_PATH = (Path(__file__).resolve().parents[1] /
                      "example_2_calibration" / "stereo_calibration.npz")


class StereoCalibration:
    """Holds calibration data and applies rectification to frame pairs."""

    def __init__(self, K1, D1, K2, D2, R1, R2, P1, P2, Q, image_size):
        self.K1, self.D1 = K1, D1
        self.K2, self.D2 = K2, D2
        self.P1, self.P2 = P1, P2
        self.Q = Q
        self.image_size = tuple(int(v) for v in image_size)

        # Rectified intrinsics live in P1: fx == fy == P1[0,0]
        self.fx = float(P1[0, 0])
        self.cx = float(P1[0, 2])
        self.cy = float(P1[1, 2])
        # P2[0,3] = -fx * baseline  (baseline in the units of T, i.e. mm
        # if the calibration square size was given in mm)
        self.baseline = float(-P2[0, 3] / P2[0, 0])

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, self.image_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, self.image_size, cv2.CV_32FC1)

    @classmethod
    def load(cls, path=DEFAULT_CALIB_PATH):
        """Load a calibration .npz saved by example_2. Raises
        FileNotFoundError if it doesn't exist."""
        data = np.load(str(path))
        return cls(data["K1"], data["D1"], data["K2"], data["D2"],
                   data["R1"], data["R2"], data["P1"], data["P2"],
                   data["Q"], data["image_size"])

    @classmethod
    def approximate(cls, image_size, fov_deg=120.0, baseline_mm=60.0):
        """Fallback when no calibration file exists: build a rough
        pinhole model from the advertised FOV and a ruler measurement
        of the lens spacing. Depth will be in the right ballpark but
        NOT trustworthy - run Example 2 for real numbers."""
        w, h = image_size
        fx = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        K = np.array([[fx, 0, w / 2.0],
                      [0, fx, h / 2.0],
                      [0, 0, 1.0]])
        D = np.zeros(5)
        R_ident = np.eye(3)
        P1 = np.hstack([K, np.zeros((3, 1))])
        P2 = P1.copy()
        P2[0, 3] = -fx * baseline_mm
        # Q assembled the same way stereoRectify would for this geometry
        Q = np.array([[1, 0, 0, -w / 2.0],
                      [0, 1, 0, -h / 2.0],
                      [0, 0, 0, fx],
                      [0, 0, 1.0 / baseline_mm, 0]])
        return cls(K, D, K, D, R_ident, R_ident, P1, P2, Q, image_size)

    def rectify(self, left_bgr, right_bgr):
        """Apply rectification so epipolar lines are horizontal rows."""
        left_r = cv2.remap(left_bgr, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_r = cv2.remap(right_bgr, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return left_r, right_r

    def depth_from_disparity(self, disparity_px):
        """Metric depth (same units as baseline) from disparity in pixels:
        Z = fx * B / d"""
        d = np.asarray(disparity_px, dtype=np.float64)
        with np.errstate(divide='ignore'):
            return np.where(d > 0, self.fx * self.baseline / d, 0.0)


def load_or_approximate(image_size, path=DEFAULT_CALIB_PATH):
    """Convenience: real calibration if available, else approximate.
    Returns (calibration, used_real_file: bool)."""
    try:
        return StereoCalibration.load(path), True
    except (FileNotFoundError, OSError):
        return StereoCalibration.approximate(image_size), False
