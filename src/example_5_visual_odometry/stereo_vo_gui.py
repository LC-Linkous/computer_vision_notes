##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/example_5_visual_odometry/stereo_vo_gui.py'
#   Example 5: stereo visual odometry - recovering the CAMERA'S OWN
#   motion through the world. This is the front-end of stereo SLAM
#   (what's missing for full SLAM is the back-end: loop closure and
#   global optimization, e.g. pose graphs / bundle adjustment).
#
#   Per-frame pipeline:
#     1. Rectify the stereo pair.
#     2. Detect ORB features in the left image, match left<->right
#        along epipolar rows -> disparity -> triangulated 3D landmark
#        for each feature (Z = fx * B / d).
#     3. Match the CURRENT left features to the PREVIOUS left features
#        (descriptor matching over time).
#     4. We now have 3D points (from the previous frame) paired with
#        2D observations (in the current frame): solvePnPRansac
#        recovers the relative camera motion.
#     5. Accumulate the relative motions into a global pose and plot
#        the trajectory top-down (X-Z plane).
#
#   Walk the camera around (slowly!) and watch the path build. Expect
#   drift - that's exactly the problem loop closure exists to solve,
#   and a good motivation for the SLAM stage of this project.
#
#   Housekeeping changes (same fixes as examples 1-4):
#     * FIXED BROKEN IMPORTS: sys.path gets 'src/' appended, so modules
#       import as 'common.*', not 'src.common.*'. The old imports
#       raised ModuleNotFoundError on launch.
#     * Camera index is a plain config variable (default 1, the stereo
#       rig on this machine). CLI arg still overrides if given.
#     * Preview is display-only and pinned to a fixed size; the VO
#       pipeline runs at full rectified resolution.
#     * Reentrancy guard + try/except around the timer handler
#       (2x ORB@1500 features + two knnMatch passes + PnP RANSAC is
#       heavy; a slow frame must skip, not queue).
#     * The trajectory plot only redraws every plot_every frames, and
#       long trajectories are DECIMATED for drawing (the full path is
#       kept for the pose/length math) - otherwise redraw cost grows
#       without bound as you walk.
#     * Reset button also clears the previous-frame landmarks, and
#       warns if the calibration resolution doesn't match the camera.
#
#   Run:  python stereo_vo_gui.py [camera_index]
#   Strongly recommends a real calibration from example_2.
##--------------------------------------------------------------------\

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import wx
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.stereo_camera import StereoCamera
from common.capture_thread import StereoCaptureThread
from common.calibration import load_or_approximate
from common.conversions import cv_to_wx_bitmap

MIN_PNP_POINTS = 15        # below this, skip the frame (motion too uncertain)
MAX_LANDMARK_DEPTH = 5000  # mm; distant points carry little parallax signal
RANSAC_REPROJ_ERR = 2.0    # px
MAX_ROW_ERROR_PX = 2.0     # epipolar row check: a left/right match must sit
                           # within this many rows to be triangulated. With
                           # good rectification ~2 is right; raising it lets
                           # VO limp along on a misaligned rig, but every
                           # extra pixel of tolerance admits mismatches and
                           # skews landmark depths - fix the calibration
                           # rather than living with a large value here.


def fit_preview(image, box_w, box_h):
    """Resize to fill a box, preserving aspect ratio. Unlike
    common.conversions.scale_to_fit this also UPSCALES, so the preview
    reaches the requested size even when the source frame is smaller."""
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=interp)


class StereoVisualOdometry:
    """Frame-to-frame stereo VO using ORB + triangulation + PnP."""

    def __init__(self, calib):
        self.calib = calib
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.K = np.array([[calib.fx, 0, calib.cx],
                           [0, calib.fx, calib.cy],
                           [0, 0, 1.0]])
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points3d = None      # aligned with prev keypoints (or NaN)
        self.pose = np.eye(4)          # camera-to-world
        self.trajectory = [self.pose[:3, 3].copy()]
        self.last_median_disp = float('nan')  # diagnostic (see triangulate)
        self.last_row_error = float('nan')    # median |vl - vr| of matches

    def reset(self):
        """Zero the pose AND forget the previous frame's landmarks, so
        the first motion after a reset is estimated fresh rather than
        against pre-reset geometry."""
        self.pose = np.eye(4)
        self.trajectory = [self.pose[:3, 3].copy()]
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points3d = None

    # ---------------- stereo triangulation ----------------
    def triangulate(self, gray_l, gray_r):
        """Detect ORB in both eyes, match along epipolar rows, and
        compute a 3D point per left keypoint (NaN where unmatched)."""
        kps_l, des_l = self.orb.detectAndCompute(gray_l, None)
        kps_r, des_r = self.orb.detectAndCompute(gray_r, None)
        n = len(kps_l)
        points3d = np.full((n, 3), np.nan)
        if des_l is None or des_r is None or n == 0 or len(kps_r) == 0:
            return kps_l, des_l, points3d

        matches = self.matcher.knnMatch(des_l, des_r, k=2)
        row_pass_disps = []   # disparities of matches that pass the row
                              # check, BEFORE the sign check (diagnostic)
        row_errors = []       # |vl - vr| of ALL ratio-passing matches:
                              # how well rectification aligns the rows
        for pair in matches:
            if len(pair) < 2:
                continue
            m, second = pair
            if m.distance > 0.75 * second.distance:   # Lowe ratio test
                continue
            ul, vl = kps_l[m.queryIdx].pt
            ur, vr = kps_r[m.trainIdx].pt
            row_errors.append(abs(vl - vr))
            # Rectified geometry: matches share a row, disparity positive
            if abs(vl - vr) > MAX_ROW_ERROR_PX:
                continue
            d = ul - ur
            row_pass_disps.append(d)
            if d <= 1.0:
                continue
            z = self.calib.fx * self.calib.baseline / d
            if z <= 0 or z > MAX_LANDMARK_DEPTH:
                continue
            x = (ul - self.calib.cx) * z / self.calib.fx
            y = (vl - self.calib.cy) * z / self.calib.fx
            points3d[m.queryIdx] = (x, y, z)
        self.last_median_disp = (float(np.median(row_pass_disps))
                                 if row_pass_disps else float('nan'))
        self.last_row_error = (float(np.median(row_errors))
                               if row_errors else float('nan'))
        return kps_l, des_l, points3d

    # ---------------- temporal step ----------------
    def step(self, gray_l, gray_r):
        """Process one rectified pair. Returns (stats_dict, kps).

        stats traces the pipeline stage by stage so a dead trajectory
        can be diagnosed from the status bar:
          keypoints -> landmarks (stereo-triangulated 3D points)
          -> temporal (matched to previous frame with valid 3D)
          -> inliers (survived PnP RANSAC)
        """
        kps, des, points3d = self.triangulate(gray_l, gray_r)
        stats = {
            'keypoints': len(kps),
            'landmarks': int(np.sum(~np.isnan(points3d[:, 0]))),
            'temporal': 0,
            'inliers': 0,
            'median_disp': self.last_median_disp,
            'row_error': self.last_row_error,
        }

        if (self.prev_descriptors is not None and des is not None
                and len(des) > 0 and len(self.prev_descriptors) > 0):
            matches = self.matcher.knnMatch(self.prev_descriptors, des, k=2)
            object_pts, image_pts = [], []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, second = pair
                if m.distance > 0.75 * second.distance:
                    continue
                p3d = self.prev_points3d[m.queryIdx]
                if np.any(np.isnan(p3d)):
                    continue
                object_pts.append(p3d)
                image_pts.append(kps[m.trainIdx].pt)
            stats['temporal'] = len(object_pts)

            if len(object_pts) >= MIN_PNP_POINTS:
                object_pts = np.array(object_pts, dtype=np.float64)
                image_pts = np.array(image_pts, dtype=np.float64)
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_pts, image_pts, self.K, None,
                    reprojectionError=RANSAC_REPROJ_ERR,
                    iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok and inliers is not None and len(inliers) >= MIN_PNP_POINTS:
                    stats['inliers'] = len(inliers)
                    R, _ = cv2.Rodrigues(rvec)
                    # T_rel maps previous-camera coords -> current-camera
                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R
                    T_rel[:3, 3] = tvec.ravel()
                    # Update camera-to-world pose
                    self.pose = self.pose @ np.linalg.inv(T_rel)
                    self.trajectory.append(self.pose[:3, 3].copy())

        self.prev_keypoints = kps
        self.prev_descriptors = des
        self.prev_points3d = points3d
        return stats, kps


class VOFrame(wx.Frame):
    def __init__(self, parent, title, camera_index=1):
        super().__init__(parent, title=title, size=(1340, 700))

        # ------------------------- config -------------------------
        self.camera_index = camera_index  # which device (see __main__)
        self.preview_w = 720              # fixed PREVIEW size (1.5x the
        self.preview_h = 540              # old 480x360); display only
        self.plot_every = 4               # redraw trajectory every Nth frame
        self.max_plot_poses = 2000        # decimate the DRAWN path beyond
                                          # this (full path kept for math)

        # ------------------------- layout -------------------------
        self.panel = wx.Panel(self)
        blank = wx.Bitmap.FromBuffer(
            self.preview_w, self.preview_h,
            np.zeros((self.preview_h, self.preview_w, 3), dtype=np.uint8))
        self.video_bitmap = wx.StaticBitmap(self.panel, bitmap=blank)

        self.fig = plt.figure(figsize=(5.4, 5.4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.panel, -1, self.fig)

        self.reset_button = wx.Button(self.panel, label="Reset Trajectory")
        self.reset_button.Bind(wx.EVT_BUTTON, self.on_reset)
        self.status = wx.StaticText(self.panel, label="")

        feeds = wx.BoxSizer(wx.HORIZONTAL)
        feeds.Add(self.video_bitmap, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        feeds.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        controls = wx.BoxSizer(wx.HORIZONTAL)
        controls.Add(self.reset_button, 0, wx.ALL, 5)
        controls.Add(self.status, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(feeds, 1, wx.EXPAND | wx.ALL, 5)
        main.Add(controls, 0, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(main)

        # ---------------- camera, calibration, state ----------------
        self.camera = StereoCamera(self.camera_index)
        warnings = []
        if not self.camera.is_opened():
            warnings.append(f"could not open camera {self.camera_index}")

        self.calib, used_real = load_or_approximate(self.camera.frame_size())
        if not used_real:
            warnings.append("approximate calibration - trajectory scale "
                            "will be wrong; run example_2 first")
        elif tuple(self.calib.image_size) != tuple(self.camera.frame_size()):
            warnings.append(
                f"calibration size {self.calib.image_size} != camera "
                f"{self.camera.frame_size()} - motion estimates will be "
                "wrong; recalibrate at this resolution")
        self._warn_prefix = ""
        if warnings:
            self._warn_prefix = "WARNING: " + "; ".join(warnings) + "\n"
            self.status.SetLabel(self._warn_prefix.strip())

        self.vo = StereoVisualOdometry(self.calib)
        self._busy = False
        self._frame_count = 0

        self.capture_thread = StereoCaptureThread(self.camera)
        self.capture_thread.start()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.timer.Start(int(1000 / 12))

    # ------------------------- callbacks -------------------------
    def update(self, event):
        if self._busy:
            return  # previous frame still processing; skip this tick
        self._busy = True
        try:
            self._process_latest_frame()
        except Exception as e:
            traceback.print_exc()
            self.status.SetLabel(f"ERROR: {e!r} (see terminal)")
            self.timer.Stop()
        finally:
            self._busy = False

    def _process_latest_frame(self):
        pair = self.capture_thread.get_latest()
        if pair is None:
            return
        left, right = pair
        left_r, right_r = self.calib.rectify(left, right)
        gray_l = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)

        stats, kps = self.vo.step(gray_l, gray_r)

        # Preview is display-only, resized (up OR down) to a fixed box
        # so it fills it and the layout never reshuffles.
        display = cv2.drawKeypoints(left_r, kps, None, color=(0, 255, 0))
        preview = fit_preview(display, self.preview_w, self.preview_h)
        self.video_bitmap.SetBitmap(cv_to_wx_bitmap(preview))

        self._frame_count += 1
        if self._frame_count % self.plot_every == 0:
            self.draw_trajectory()

        traj = self.vo.trajectory
        dist_m = 0.0
        if len(traj) > 1:
            arr = np.array(traj)
            dist_m = float(np.sum(np.linalg.norm(np.diff(arr, axis=0),
                                                 axis=1))) / 1000.0

        # Pipeline trace: whichever number collapses to ~0 is the stage
        # that's failing.
        text = (f"kps: {stats['keypoints']}  landmarks: {stats['landmarks']}  "
                f"temporal: {stats['temporal']}  inliers: {stats['inliers']}  |  "
                f"poses: {len(traj)}  path: {dist_m:.2f} m")
        if stats['median_disp'] == stats['median_disp']:  # not NaN
            if stats['median_disp'] < 0:
                text += ("  |  median disparity NEGATIVE - left/right eyes "
                         "may be SWAPPED in StereoCamera")
        elif stats['keypoints'] > 0:
            if stats['row_error'] == stats['row_error']:  # not NaN
                text += (f"  |  rows misaligned by ~{stats['row_error']:.0f} px "
                         f"(need <{MAX_ROW_ERROR_PX:g}) - rectification is "
                         "not working")
            else:
                text += "  |  no stereo matches at all - texture? focus?"
        self.status.SetLabel(self._warn_prefix + text)

    def draw_trajectory(self):
        traj = np.array(self.vo.trajectory)
        # Decimate long paths for DRAWING only - redrawing every pose
        # forever makes the plot cost grow without bound. Always keep
        # the final pose so the red marker is exact.
        if len(traj) > self.max_plot_poses:
            step = len(traj) // self.max_plot_poses + 1
            traj = np.vstack([traj[::step], traj[-1]])
        self.ax.clear()
        self.ax.plot(traj[:, 0], traj[:, 2], 'b-', linewidth=1)
        self.ax.plot(traj[-1, 0], traj[-1, 2], 'ro', markersize=6)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Z (mm)')
        self.ax.set_title('Estimated camera trajectory (top-down)')
        self.ax.set_aspect('equal', adjustable='datalim')
        self.canvas.draw()

    def on_reset(self, event):
        self.vo.reset()
        self.draw_trajectory()

    def on_close(self, event):
        self.timer.Stop()
        self.capture_thread.stop()
        self.capture_thread.join(timeout=2.0)
        self.camera.release()
        self.Destroy()


if __name__ == '__main__':
    # Camera device index. The stereo rig enumerates at 1 on this
    # machine; pass a different index on the command line to override.
    DEFAULT_CAMERA_INDEX = 1
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CAMERA_INDEX
    app = wx.App()
    frame = VOFrame(None, 'Example 5: Stereo Visual Odometry', cam_idx)
    frame.Show()
    app.MainLoop()