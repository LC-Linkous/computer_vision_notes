##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/example_1_orb_depth/gui_frame.py'
#   Example 1: live stereo feed + ORB keypoints + naive depth scatter.
#
#   Changes from previous version (fixes "blank preview" freeze):
#     * Camera I/O moved to StereoCaptureThread. The wx.Timer handler
#       never blocks on VideoCapture.read(); it just grabs the newest
#       pair, so the UI thread stays responsive and paint events run.
#     * Requests the camera's low-res side-by-side mode (1280x480 ->
#       two 640x480 eyes) and additionally downscales with
#       scale_to_fit() in case the camera ignores the request.
#       StereoBM + 2x ORB on 1920x1080 eyes at 24fps was saturating
#       the UI thread, which is why bitmaps resized but never painted.
#     * Reentrancy guard: if a frame is still being processed when the
#       timer fires again, the new tick is skipped instead of queued.
#     * The matplotlib 3D scatter (the slowest single draw call) only
#       redraws every plot_every frames.
#     * update() body wrapped in try/except; errors go to the status
#       box AND stderr instead of being swallowed by wx.
#     * FPS counter in the status box so you can see it's alive.
#
#   Run:  python gui_frame.py [camera_index]
##--------------------------------------------------------------------\

import sys
import time
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
from common.conversions import cv_to_wx_bitmap, scale_to_fit

from example_1_orb_depth.point_cloud import PointCloud


class GFrame(wx.Frame):
    def __init__(self, parent, title, camera_index=1):
        super().__init__(parent, title=title, size=(1300, 575))

        # ------------------------- config -------------------------
        self.camera_index = camera_index  # which device to open (see __main__)
        self.frame_interval_ms = int(1000 / 24)
        self.show_orb = True
        self.max_plot_points = 300   # cap scatter size to keep redraws fast
        self.plot_every = 5          # redraw 3D scatter every Nth frame
        self.proc_max_w = 640        # per-eye PROCESSING size cap
        self.proc_max_h = 480
        self.preview_w = 320         # fixed PREVIEW size; display only,
        self.preview_h = 240         # processing still runs at proc_max
        # Heat map: exponentially-decayed accumulation of ORB keypoint
        # hits. Hot = plenty of trackable texture; cold = feature-based
        # vision is blind there (blank walls, glare, repeating pattern).
        self.heat_decay = 0.95       # per-frame memory (0.95 @ ~20fps
                                     # gives a half-life of ~0.7 s)
        self.heat_sigma = 11         # Gaussian blur radius in px
        self.heat_alpha = 0.55       # overlay opacity on the live frame
        self.point_cloud = PointCloud()

        # ------------------------- layout -------------------------
        self.panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left side: stereo feeds + status text
        self.camera_panel = wx.Panel(self.panel)
        self.left_panel = wx.Panel(self.camera_panel)
        self.right_panel = wx.Panel(self.camera_panel)

        default_bitmap = wx.Bitmap.FromBuffer(
            320, 240, np.zeros((240, 320, 3), dtype=np.uint8))
        self.left_bitmap = wx.StaticBitmap(self.left_panel, bitmap=default_bitmap)
        self.right_bitmap = wx.StaticBitmap(self.right_panel, bitmap=default_bitmap)

        left_sizer = wx.BoxSizer(wx.VERTICAL)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(self.left_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(self.right_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        self.left_panel.SetSizer(left_sizer)
        self.right_panel.SetSizer(right_sizer)

        top_camera_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_camera_sizer.Add(self.left_panel, 1, wx.EXPAND | wx.ALL, 5)
        top_camera_sizer.Add(self.right_panel, 1, wx.EXPAND | wx.ALL, 5)

        self.status_box = wx.TextCtrl(self.camera_panel, value="",
                                      style=wx.TE_MULTILINE | wx.TE_READONLY)
        bottom_textbox_sizer = wx.BoxSizer(wx.VERTICAL)
        bottom_textbox_sizer.Add(self.status_box, 1, wx.EXPAND | wx.ALL, 5)

        camera_sizer = wx.BoxSizer(wx.VERTICAL)
        camera_sizer.Add(top_camera_sizer, 2, wx.EXPAND | wx.ALL, 5)
        camera_sizer.Add(bottom_textbox_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.camera_panel.SetSizer(camera_sizer)

        # Right side: data notebook
        self.data_notebook = wx.Notebook(self.panel)

        # Point cloud page (matplotlib 3D)
        self.pointcloud_panel = wx.Panel(self.data_notebook)
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.pointcloud_panel, -1, self.fig)
        pointcloud_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pointcloud_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.pointcloud_panel.SetSizer(pointcloud_sizer)

        # Heat map page (keypoint density overlaid on the live left feed)
        self.heatmap_page = wx.Panel(self.data_notebook)
        heatmap_sizer = wx.BoxSizer(wx.VERTICAL)
        self.heatmap_bitmap = wx.StaticBitmap(
            self.heatmap_page,
            bitmap=wx.Bitmap.FromBuffer(320, 240,
                                        np.zeros((240, 320, 3), dtype=np.uint8)))
        heatmap_sizer.Add(self.heatmap_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        self.heatmap_page.SetSizer(heatmap_sizer)

        # Disparity map page
        self.disparity_page = wx.Panel(self.data_notebook)
        disparity_sizer = wx.BoxSizer(wx.VERTICAL)
        self.disparity_bitmap = wx.StaticBitmap(
            self.disparity_page,
            bitmap=wx.Bitmap.FromBuffer(320, 240,
                                        np.zeros((240, 320, 3), dtype=np.uint8)))
        disparity_sizer.Add(self.disparity_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        self.disparity_page.SetSizer(disparity_sizer)

        self.data_notebook.AddPage(self.pointcloud_panel, "Point Cloud")
        self.data_notebook.AddPage(self.heatmap_page, "Heat Map")
        self.data_notebook.AddPage(self.disparity_page, "Disparity Map")

        top_right_sizer = wx.BoxSizer(wx.VERTICAL)
        top_right_sizer.Add(self.data_notebook, 1, wx.EXPAND | wx.ALL, 5)

        top_sizer.Add(self.camera_panel, 1, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(top_right_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Bottom: buttons
        self.buttons_panel = wx.Panel(self.panel)
        self.start_button = wx.Button(self.buttons_panel, label="Start Video")
        self.stop_button = wx.Button(self.buttons_panel, label="Stop Video")
        self.start_button.Bind(wx.EVT_BUTTON, self.start_video)
        self.stop_button.Bind(wx.EVT_BUTTON, self.stop_video)
        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        buttons_sizer.Add(self.start_button, 0, wx.ALL, 5)
        buttons_sizer.Add(self.stop_button, 0, wx.ALL, 5)
        self.buttons_panel.SetSizer(buttons_sizer)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(top_sizer, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.buttons_panel, 0, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(main_sizer)

        # ------------------------- camera -------------------------
        self.camera = None
        self.capture_thread = None
        self.is_playing = False
        self._busy = False           # reentrancy guard for update()
        self._heat_acc = None        # keypoint density accumulator
        self._frame_count = 0
        self._fps_t0 = time.monotonic()
        self._fps_frames = 0

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.open_camera()

    # ------------------------- camera management -------------------------
    def open_camera(self):
        self.close_camera()
        # Request the small side-by-side mode (2 x 640x480); if the
        # camera ignores it, scale_to_fit() keeps processing cheap.
        self.camera = StereoCamera(self.camera_index, width=1280, height=480)
        if not self.camera.is_opened():
            self.update_status_text(
                f"WARNING: could not open camera {self.camera_index}")
            self.camera.release()
            self.camera = None
            return
        w, h = self.camera.frame_size()
        self.update_status_text(
            f"camera {self.camera_index} open, per-eye size {w}x{h}")

        self.capture_thread = StereoCaptureThread(self.camera)
        self.capture_thread.start()

    def close_camera(self):
        self.is_playing = False
        self.timer.Stop()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    # ------------------------- callbacks -------------------------
    def update(self, event):
        if not self.is_playing:
            self.timer.Stop()
            return
        if self._busy:
            return  # previous frame still processing; skip this tick
        self._busy = True
        try:
            self._process_latest_frame()
        except Exception as e:
            traceback.print_exc()
            self.update_status_text(f"ERROR in update: {e!r}")
            self.stop_video(None)
        finally:
            self._busy = False

    def _process_latest_frame(self):
        if self.capture_thread is None:
            return
        pair = self.capture_thread.get_latest()
        if pair is None:
            return  # no frame captured yet
        left, right = pair

        # Downscale before any processing; StereoBM/ORB on full-res
        # eyes is what was starving the UI thread.
        left = scale_to_fit(left, self.proc_max_w, self.proc_max_h)
        right = scale_to_fit(right, self.proc_max_w, self.proc_max_h)

        left_annot, right_annot, disparity_vis, coords, kp_pixels = \
            self.point_cloud.detect_keypoints(left, right,
                                              draw_keypoints=self.show_orb)

        heat_overlay = self.update_heatmap(left, kp_pixels)

        # Previews are display-only: shrink to a fixed size so the
        # StaticBitmaps never grow and reshuffle the layout.
        # (SetBitmap adopts the bitmap's size, so without this the
        # panels expand to the full processing resolution.)
        left_prev = scale_to_fit(left_annot, self.preview_w, self.preview_h)
        right_prev = scale_to_fit(right_annot, self.preview_w, self.preview_h)
        disp_prev = scale_to_fit(disparity_vis, self.preview_w, self.preview_h)
        heat_prev = scale_to_fit(heat_overlay, self.preview_w, self.preview_h)

        self.left_bitmap.SetBitmap(cv_to_wx_bitmap(left_prev))
        self.right_bitmap.SetBitmap(cv_to_wx_bitmap(right_prev))
        self.disparity_bitmap.SetBitmap(cv_to_wx_bitmap(disp_prev))
        self.heatmap_bitmap.SetBitmap(cv_to_wx_bitmap(heat_prev))

        self._frame_count += 1
        if self._frame_count % self.plot_every == 0:
            self.graph_point_cloud(coords)

        # Lightweight FPS readout every ~2s so we know it's alive
        self._fps_frames += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 2.0:
            fps = self._fps_frames / (now - self._fps_t0)
            self.update_status_text(f"displaying at {fps:.1f} fps, "
                                    f"{len(coords)} keypoints with depth")
            self._fps_t0 = now
            self._fps_frames = 0

    def start_video(self, event):
        if self.camera is None or self.capture_thread is None:
            # Camera failed to open earlier — retry instead of erroring.
            self.open_camera()
            if self.camera is None:
                self.update_status_text(
                    "could not open camera — is another app using it?")
                return
        self.is_playing = True
        self.timer.Start(self.frame_interval_ms)
        self.update_status_text("starting video")

    def stop_video(self, event):
        self.is_playing = False
        self.update_status_text("stopping video")

    def on_close(self, event):
        self.close_camera()
        self.Destroy()
        wx.Exit()

    # ------------------------- helpers -------------------------
    def update_heatmap(self, frame_bgr, keypoint_pixels):
        """Keypoint-density heat map, blended over the live left frame.

        Each frame: decay the accumulator (exponential forgetting), splat
        a hit at every ORB keypoint, blur to spread hits into regions,
        normalize, colormap, and alpha-blend onto the frame. Hot zones
        are where the scene has trackable texture; cold zones are where
        feature-based vision (and Example 4's tracker) goes blind.
        Swap the splat line to accumulate disparity validity instead and
        this same machinery maps where STEREO MATCHING works.
        """
        h, w = frame_bgr.shape[:2]
        if self._heat_acc is None or self._heat_acc.shape != (h, w):
            self._heat_acc = np.zeros((h, w), dtype=np.float32)

        self._heat_acc *= self.heat_decay
        for x, y in keypoint_pixels:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                self._heat_acc[yi, xi] += 1.0

        blurred = cv2.GaussianBlur(self._heat_acc, (0, 0), self.heat_sigma)
        peak = blurred.max()
        if peak > 0:
            norm = np.uint8(np.clip(blurred / peak, 0, 1) * 255)
        else:
            norm = np.zeros((h, w), dtype=np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        return cv2.addWeighted(frame_bgr, 1.0 - self.heat_alpha,
                               colored, self.heat_alpha, 0)

    def update_status_text(self, text):
        if text is None:
            return
        cur_time = time.strftime("%H:%M:%S", time.localtime())
        self.status_box.AppendText(f"[{cur_time}] {text}\n")

    def graph_point_cloud(self, coords):
        coords = np.array(coords)
        if coords.ndim != 2 or len(coords) < 3:
            return  # not enough points this frame; keep last plot
        if len(coords) > self.max_plot_points:
            idx = np.random.choice(len(coords), self.max_plot_points,
                                   replace=False)
            coords = coords[idx]
        self.ax.clear()
        self.ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=coords[:, 2], cmap='plasma', s=8)
        self.ax.set_xlabel('X (px)')
        self.ax.set_ylabel('Y (px)')
        self.ax.set_zlabel('Pseudo-depth (1/disparity)')
        self.canvas.draw()


if __name__ == '__main__':
    # Camera device index. The stereo rig enumerates at 1 on this
    # machine; pass a different index on the command line to override.
    DEFAULT_CAMERA_INDEX = 1
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CAMERA_INDEX
    app = wx.App()
    frame = GFrame(None, 'Example 1: ORB + Naive Stereo Depth', cam_idx)
    frame.Show()
    app.MainLoop()