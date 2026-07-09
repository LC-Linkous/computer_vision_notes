##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/example_3_rectified_depth/rectified_depth_gui.py'
#   Example 3: metric depth from a CALIBRATED, RECTIFIED stereo pair.
#
#   Differences from Example 1:
#     * Frames are rectified first, so block matching searches along
#       true epipolar lines (horizontal rows). Toggle the green guide
#       lines to verify: matching features should sit on the same row.
#     * StereoSGBM (semi-global matching) replaces StereoBM - slower
#       but much denser and cleaner disparity.
#     * cv2.reprojectImageTo3D + the Q matrix turn disparity into
#       (X, Y, Z) in real units (mm, if Example 2 used mm squares).
#     * Capture runs on a background thread (common/capture_thread.py)
#       because SGBM is heavy enough to stall a UI-thread pipeline.
#
#   Housekeeping changes (same fixes as examples 1/2/4):
#     * FIXED BROKEN IMPORTS: sys.path gets 'src/' appended, so modules
#       import as 'common.*', not 'src.common.*'. The old imports
#       raised ModuleNotFoundError on launch.
#     * Camera index is a plain config variable (default 1, the stereo
#       rig on this machine). CLI arg still overrides if given.
#     * Previews are display-only and pinned to a fixed size so the
#       StaticBitmaps never grow and reshuffle the layout. Unlike
#       common.conversions.scale_to_fit (shrink-only), fit_preview()
#       below also UPSCALES, so the preview fills its box even when
#       the camera frame is smaller than it. SGBM still runs at full
#       rectified resolution.
#     * Reentrancy guard + try/except so a slow SGBM frame can't pile
#       up timer events or die silently.
#     * The 3D cloud (reprojectImageTo3D + matplotlib scatter, by far
#       the slowest part after SGBM) only refreshes every plot_every
#       frames; the video and disparity views update every frame.
#     * Warns if the calibration resolution doesn't match the camera.
#
#   Run:  python rectified_depth_gui.py [camera_index]
#   Requires example_2_calibration/stereo_calibration.npz; falls back
#   to an approximate model (with a loud warning) if it is missing.
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


def make_sgbm(num_disparities=128, block_size=5):
    """StereoSGBM with the standard P1/P2 smoothness settings."""
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,   # must be divisible by 16
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def fit_preview(image, box_w, box_h):
    """Resize to fill a box, preserving aspect ratio. Unlike
    common.conversions.scale_to_fit this also UPSCALES, so previews
    reach the requested size even when the source frame is smaller."""
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=interp)


class DepthFrame(wx.Frame):
    def __init__(self, parent, title, camera_index=1):
        super().__init__(parent, title=title, size=(1800, 700))

        # ------------------------- config -------------------------
        self.camera_index = camera_index  # which device (see __main__)
        self.max_plot_points = 1500
        self.max_depth_mm = 3000.0   # clip the cloud at 3 m for display
        self.plot_every = 4          # refresh 3D cloud every Nth frame
        self.preview_w = 504         # fixed PREVIEW size (1.2x the old
        self.preview_h = 384         # 420x320); display only

        # ------------------------- layout -------------------------
        self.panel = wx.Panel(self)

        blank = wx.Bitmap.FromBuffer(
            self.preview_w, self.preview_h,
            np.zeros((self.preview_h, self.preview_w, 3), dtype=np.uint8))
        self.left_bitmap = wx.StaticBitmap(self.panel, bitmap=blank)
        self.disparity_bitmap = wx.StaticBitmap(self.panel, bitmap=blank)

        self.fig = plt.figure(figsize=(4.5, 4.5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.panel, -1, self.fig)

        self.lines_checkbox = wx.CheckBox(self.panel,
                                          label="Show epipolar guide lines")
        self.status = wx.StaticText(self.panel, label="")

        feeds = wx.BoxSizer(wx.HORIZONTAL)
        feeds.Add(self.left_bitmap, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        feeds.Add(self.disparity_bitmap, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        feeds.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        controls = wx.BoxSizer(wx.HORIZONTAL)
        controls.Add(self.lines_checkbox, 0, wx.ALL, 5)
        controls.Add(self.status, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(feeds, 1, wx.EXPAND | wx.ALL, 5)
        main.Add(controls, 0, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(main)

        # ------------------------- camera + calib -------------------------
        self.camera = StereoCamera(self.camera_index)
        warnings = []
        if not self.camera.is_opened():
            warnings.append(f"could not open camera {self.camera_index}")

        self.calib, used_real = load_or_approximate(self.camera.frame_size())
        if not used_real:
            warnings.append("no calibration file found - using APPROXIMATE "
                            "model; run example_2 for metric depth")
        elif tuple(self.calib.image_size) != tuple(self.camera.frame_size()):
            warnings.append(
                f"calibration size {self.calib.image_size} != camera "
                f"{self.camera.frame_size()} - depth will be wrong; "
                "recalibrate at this resolution")

        if warnings:
            self.status.SetLabel("WARNING: " + "; ".join(warnings))
        else:
            self.status.SetLabel(
                f"Calibrated. fx={self.calib.fx:.1f}px  "
                f"baseline={self.calib.baseline:.1f}mm")

        self.sgbm = make_sgbm()
        self._busy = False
        self._frame_count = 0

        self.capture_thread = StereoCaptureThread(self.camera)
        self.capture_thread.start()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.timer.Start(int(1000 / 12))  # SGBM-friendly UI rate

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

        gray_l = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)

        disparity = self.sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0

        self.update_views(left_r, disparity)

        # The 3D cloud is the slowest stage after SGBM itself
        # (reprojectImageTo3D over the full frame + a matplotlib 3D
        # scatter), so only refresh it every plot_every frames.
        self._frame_count += 1
        if self._frame_count % self.plot_every == 0:
            # points[y, x] = (X, Y, Z) in calibration units
            points_3d = cv2.reprojectImageTo3D(disparity, self.calib.Q)
            self.update_cloud(points_3d, disparity, left_r)

    def update_views(self, left_r, disparity):
        display = left_r.copy()
        if self.lines_checkbox.GetValue():
            for y in range(0, display.shape[0], 40):
                cv2.line(display, (0, y), (display.shape[1], y), (0, 255, 0), 1)
        disp_vis = cv2.normalize(disparity, None, 0, 255,
                                 cv2.NORM_MINMAX, cv2.CV_8U)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)

        # Previews are display-only, resized (up OR down) to a fixed
        # box so they fill it and the layout never reshuffles.
        left_prev = fit_preview(display, self.preview_w, self.preview_h)
        disp_prev = fit_preview(disp_vis, self.preview_w, self.preview_h)
        self.left_bitmap.SetBitmap(cv_to_wx_bitmap(left_prev))
        self.disparity_bitmap.SetBitmap(cv_to_wx_bitmap(disp_prev))

    def update_cloud(self, points_3d, disparity, left_r):
        mask = (disparity > 1.0) & np.isfinite(points_3d[:, :, 2])
        mask &= (points_3d[:, :, 2] > 0) & (points_3d[:, :, 2] < self.max_depth_mm)
        pts = points_3d[mask]
        colors = cv2.cvtColor(left_r, cv2.COLOR_BGR2RGB)[mask] / 255.0
        if len(pts) < 10:
            return
        if len(pts) > self.max_plot_points:
            idx = np.random.choice(len(pts), self.max_plot_points, replace=False)
            pts, colors = pts[idx], colors[idx]

        self.ax.clear()
        # Camera convention: X right, Y down, Z forward. Plot Z forward,
        # -Y up so the cloud appears upright.
        self.ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=colors, s=2)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Z depth (mm)')
        self.ax.set_zlabel('Height (mm)')
        self.ax.set_ylim(0, self.max_depth_mm)
        self.canvas.draw()

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
    frame = DepthFrame(None, 'Example 3: Rectified Metric Depth (SGBM)', cam_idx)
    frame.Show()
    app.MainLoop()