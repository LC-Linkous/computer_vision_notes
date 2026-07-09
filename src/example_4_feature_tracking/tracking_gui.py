##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/example_4_feature_tracking/tracking_gui.py'
#   Example 4: tracking features OVER TIME (not just across the stereo
#   pair) and fusing the tracks with stereo depth.
#
#   New concepts vs Example 3:
#     * Shi-Tomasi corners (cv2.goodFeaturesToTrack) seed the tracker.
#     * Lucas-Kanade pyramidal optical flow (cv2.calcOpticalFlowPyrLK)
#       follows each corner from frame t to frame t+1 in the LEFT eye.
#     * Each tracked point also gets a metric depth by sampling the
#       SGBM disparity map at its location -> every track is a little
#       3D trajectory. This temporal association is the missing
#       ingredient between "depth map" and "visual odometry/SLAM".
#
#   Display: left feed with colored trails (color = depth), and a
#   matplotlib top-down (X-Z) view of tracked points.
#
#   Housekeeping changes (same fixes as example 1's gui_frame.py):
#     * Camera index is a plain config variable (default 1, the stereo
#       rig on this machine). CLI arg still overrides if given.
#     * Preview is display-only and pinned to a fixed size so the
#       StaticBitmap never grows and reshuffles the layout. Tracking
#       still runs at full rectified resolution.
#     * Reentrancy guard + try/except so a slow or failing frame can't
#       pile up timer events or die silently.
#     * The matplotlib top-down view (slowest draw call) only redraws
#       every plot_every frames; the video preview updates every frame.
#
#   Run:  python tracking_gui.py [camera_index]
##--------------------------------------------------------------------\

import sys
import time
import traceback
from collections import deque
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
from common.conversions import cv_to_wx_bitmap, scale_to_fit

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           30, 0.01))
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01,
                      minDistance=12, blockSize=7)
TRAIL_LENGTH = 15          # frames of history drawn per track
REDETECT_BELOW = 80        # re-seed corners when track count drops


class Track:
    """One feature followed over time: 2D trail + latest metric depth."""
    def __init__(self, point):
        self.points = deque(maxlen=TRAIL_LENGTH)
        self.points.append(point)
        self.depth_mm = 0.0

    @property
    def latest(self):
        return self.points[-1]


class TrackingFrame(wx.Frame):
    def __init__(self, parent, title, camera_index=1):
        super().__init__(parent, title=title, size=(1250, 600))

        # ------------------------- config -------------------------
        self.camera_index = camera_index  # which device (see __main__)
        self.preview_w = 600              # fixed PREVIEW size; display
        self.preview_h = 600              # only, tracking runs full res
        self.plot_every = 5               # redraw top-down view every Nth frame

        # ------------------------- layout -------------------------
        self.panel = wx.Panel(self)
        blank = wx.Bitmap.FromBuffer(
            self.preview_w, self.preview_h,
            np.zeros((self.preview_h, self.preview_w, 3), dtype=np.uint8))
        self.video_bitmap = wx.StaticBitmap(self.panel, bitmap=blank)

        self.fig = plt.figure(figsize=(4.5, 4.5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.panel, -1, self.fig)

        self.status = wx.StaticText(self.panel, label="")

        feeds = wx.BoxSizer(wx.HORIZONTAL)
        feeds.Add(self.video_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        feeds.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(feeds, 1, wx.EXPAND | wx.ALL, 5)
        main.Add(self.status, 0, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(main)

        # ---------------- camera, calibration, state ----------------
        self.camera = StereoCamera(self.camera_index)
        warnings = []
        if not self.camera.is_opened():
            warnings.append(f"could not open camera {self.camera_index}")

        self.calib, used_real = load_or_approximate(self.camera.frame_size())
        if not used_real:
            warnings.append("approximate calibration in use - "
                            "run example_2 for metric depth")
        elif tuple(self.calib.image_size) != tuple(self.camera.frame_size()):
            warnings.append(
                f"calibration size {self.calib.image_size} != camera "
                f"{self.camera.frame_size()} - depth will be wrong; "
                "recalibrate at this resolution")
        if warnings:
            self.status.SetLabel("WARNING: " + "; ".join(warnings))

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=96, blockSize=5,
            P1=8 * 3 * 25, P2=32 * 3 * 25, disp12MaxDiff=1,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)

        self.prev_gray = None
        self.tracks = []
        self._busy = False
        self._frame_count = 0

        self.capture_thread = StereoCaptureThread(self.camera)
        self.capture_thread.start()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.timer.Start(int(1000 / 15))

    # ------------------------- pipeline -------------------------
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
        gray = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)

        # 1. advance existing tracks with optical flow
        if self.prev_gray is not None and self.tracks:
            prev_pts = np.float32([t.latest for t in self.tracks]).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None, **LK_PARAMS)
            # forward-backward check: flow back and require consistency
            back_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.prev_gray, next_pts, None, **LK_PARAMS)
            fb_error = np.abs(prev_pts - back_pts).reshape(-1, 2).max(axis=1)

            surviving = []
            for track, new_pt, ok, err in zip(self.tracks,
                                              next_pts.reshape(-1, 2),
                                              status.ravel(), fb_error):
                if ok and err < 1.0 and self.in_bounds(new_pt, gray.shape):
                    track.points.append(tuple(new_pt))
                    surviving.append(track)
            self.tracks = surviving

        # 2. re-seed corners if we are running low, away from live tracks
        if len(self.tracks) < REDETECT_BELOW:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for t in self.tracks:
                cv2.circle(mask, tuple(int(v) for v in t.latest), 10, 0, -1)
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)
            if corners is not None:
                for c in corners.reshape(-1, 2):
                    self.tracks.append(Track(tuple(c)))

        # 3. attach metric depth to every live track via the disparity map
        gray_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)
        disparity = self.sgbm.compute(gray, gray_r).astype(np.float32) / 16.0
        for t in self.tracks:
            x, y = int(t.latest[0]), int(t.latest[1])
            if 0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]:
                d = disparity[y, x]
                if d > 1.0:
                    t.depth_mm = self.calib.fx * self.calib.baseline / d

        self.prev_gray = gray
        self._frame_count += 1
        self.draw(left_r)

    @staticmethod
    def in_bounds(pt, shape):
        return 0 <= pt[0] < shape[1] and 0 <= pt[1] < shape[0]

    # ------------------------- display -------------------------
    def draw(self, left_r):
        display = left_r.copy()
        for t in self.tracks:
            if t.depth_mm <= 0:
                continue
            # near = warm, far = cool (clipped to 3 m)
            ratio = min(t.depth_mm / 3000.0, 1.0)
            color = (int(255 * ratio), 64, int(255 * (1 - ratio)))  # BGR
            pts = np.int32(t.points)
            cv2.polylines(display, [pts], False, color, 1)
            cv2.circle(display, tuple(pts[-1]), 3, color, -1)

        # Preview is display-only: pin to a fixed size so the
        # StaticBitmap never grows and reshuffles the layout.
        preview = scale_to_fit(display, self.preview_w, self.preview_h)
        self.video_bitmap.SetBitmap(cv_to_wx_bitmap(preview))

        # top-down map (X lateral vs Z depth) is the slowest draw call;
        # only redraw it every plot_every frames
        if self._frame_count % self.plot_every == 0:
            xs, zs = [], []
            for t in self.tracks:
                if t.depth_mm > 0:
                    x_px = t.latest[0]
                    z = t.depth_mm
                    xs.append((x_px - self.calib.cx) * z / self.calib.fx)
                    zs.append(z)
            self.ax.clear()
            if xs:
                self.ax.scatter(xs, zs, c=zs, cmap='plasma_r', s=8)
            self.ax.set_xlabel('X lateral (mm)')
            self.ax.set_ylabel('Z depth (mm)')
            self.ax.set_ylim(0, 3000)
            self.ax.set_title('Top-down view of tracked features')
            self.canvas.draw()

        self.status.SetLabel(f"live tracks: {len(self.tracks)}")

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
    frame = TrackingFrame(None, 'Example 4: Temporal Feature Tracking + Depth',
                          cam_idx)
    frame.Show()
    app.MainLoop()