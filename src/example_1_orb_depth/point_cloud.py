##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/example_1_orb_depth/point_cloud.py'
#   ORB keypoint detection + naive (UNCALIBRATED) depth from StereoBM.
#
#   Cleaned-up version of the original prototype. Changes from v0:
#     * Works directly on OpenCV (numpy) frames. Bitmap conversion now
#       lives in common/conversions.py and happens once, at the GUI edge.
#     * StereoBM output is divided by 16 (it returns fixed-point
#       disparity scaled by 16) before being used for depth.
#     * Detector and matcher are created once in __init__ instead of
#       per frame.
#     * Grayscale conversion happens once instead of twice.
#
#   IMPORTANT LIMITATION: without calibration/rectification the
#   "depth" here is only a relative quantity (proportional to
#   1/disparity). Example 2 + 3 fix this properly.
##--------------------------------------------------------------------\

import cv2
import numpy as np


class PointCloud:
    def __init__(self, num_disparities=64, block_size=15, n_features=500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.stereo = cv2.StereoBM_create(numDisparities=num_disparities,
                                          blockSize=block_size)

    def detect_keypoints(self, left_bgr, right_bgr, draw_keypoints=True):
        """Detect ORB keypoints in the left image, compute a block-matching
        disparity map, and associate each keypoint with a pseudo-depth.

        Returns:
            left_annotated  - left frame with keypoints drawn (BGR)
            right_annotated - right frame with keypoints drawn (BGR)
            disparity_vis   - normalized disparity map for display (uint8)
            coordinates     - list of [x_px, y_px, pseudo_depth]
            keypoint_pixels - list of (x_px, y_px) for ALL left keypoints,
                              including those with no valid disparity
                              (used by the GUI's keypoint-density heat map)
        """
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        keypoints_left, _ = self.orb.detectAndCompute(left_gray, None)
        keypoints_right, _ = self.orb.detectAndCompute(right_gray, None)

        # StereoBM returns disparity * 16 as int16 fixed point
        disparity = self.stereo.compute(left_gray, right_gray)
        disparity = disparity.astype(np.float32) / 16.0

        disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)

        coordinates = self.relate_keypoints_to_depth(keypoints_left, disparity)
        keypoint_pixels = [kp.pt for kp in keypoints_left]

        if draw_keypoints:
            left_annotated = cv2.drawKeypoints(left_bgr, keypoints_left, None,
                                               color=(0, 255, 0))
            right_annotated = cv2.drawKeypoints(right_bgr, keypoints_right,
                                                None, color=(0, 255, 0))
        else:
            left_annotated, right_annotated = left_bgr, right_bgr

        return (left_annotated, right_annotated, disparity_vis,
                coordinates, keypoint_pixels)

    def relate_keypoints_to_depth(self, keypoints, disparity):
        """Map each keypoint to a pseudo-depth value (1/disparity).

        Until the rig is calibrated (Example 2) this is only ordinal:
        bigger means farther, but the units are not meters.
        """
        coordinates = []
        h, w = disparity.shape
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if not (0 <= x < w and 0 <= y < h):
                continue
            d = disparity[y, x]
            if d > 0:
                pseudo_depth = 1.0 / d
                coordinates.append([kp.pt[0], kp.pt[1], pseudo_depth])
        return coordinates