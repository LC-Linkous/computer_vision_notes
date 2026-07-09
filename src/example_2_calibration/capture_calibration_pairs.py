##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/example_2_calibration/capture_calibration_pairs.py'
#   Example 2a: capture synchronized left/right chessboard images.
#
#   Print a chessboard pattern (default: 9x6 INNER corners, e.g. the
#   classic OpenCV pattern), tape it to something rigid, and capture
#   15-25 pairs at different distances, angles, and positions in the
#   frame (especially the corners of the image, where distortion is
#   strongest).
#
#   A green overlay confirms the board is detected in BOTH eyes before
#   you save a pair - only pairs detected in both are worth keeping.
#
#   Housekeeping changes (same fixes as example 1's gui_frame.py):
#     * Camera index is a plain config variable (default 1, the stereo
#       rig on this machine). CLI arg still overrides if given.
#     * Camera I/O moved to StereoCaptureThread so the timer handler
#       never blocks on VideoCapture.read().
#     * Chessboard DETECTION runs on a downscaled copy (it's an
#       expensive call, and here it's only a gate + overlay).
#       SAVED IMAGES ARE ALWAYS THE RAW FULL-RESOLUTION FRAMES --
#       they define the calibration's image_size, which must match
#       the resolution the later examples run at.
#     * Previews are display-only and pinned to a fixed size so the
#       StaticBitmaps never grow and reshuffle the layout.
#     * Reentrancy guard + try/except around the timer handler.
#
#   Run:  python capture_calibration_pairs.py [camera_index]
#   Output: ./calibration_images/left_NN.png, right_NN.png
##--------------------------------------------------------------------\

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import wx

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.stereo_camera import StereoCamera
from common.capture_thread import StereoCaptureThread
from common.conversions import cv_to_wx_bitmap, scale_to_fit

# Inner corner count (columns, rows). MUST match calibrate_stereo.py.
#   (7, 7) = a REAL 8x8-square chessboard
#   (9, 6) = the classic printed OpenCV 10x7-square pattern
# NOTE for real chessboards: 7x7 looks identical rotated 180 degrees,
# which can flip corner ordering between captures. Keep the board in
# the same orientation for every pair (or tape a marker on one corner).
CHESSBOARD = (7, 7)
OUTPUT_DIR = Path(__file__).resolve().parent / "calibration_images"


class CaptureFrame(wx.Frame):
    def __init__(self, parent, title, camera_index=1):
        super().__init__(parent, title=title, size=(1100, 520))

        # ------------------------- config -------------------------
        self.camera_index = camera_index  # which device (see __main__)
        self.detect_max_w = 640    # chessboard detection runs at this
        self.detect_max_h = 480    # size; saved images stay full-res
        self.preview_w = 480       # fixed PREVIEW size, display only
        self.preview_h = 360

        # ------------------------- layout -------------------------
        self.panel = wx.Panel(self)
        blank = wx.Bitmap.FromBuffer(
            self.preview_w, self.preview_h,
            np.zeros((self.preview_h, self.preview_w, 3), dtype=np.uint8))
        self.left_bitmap = wx.StaticBitmap(self.panel, bitmap=blank)
        self.right_bitmap = wx.StaticBitmap(self.panel, bitmap=blank)

        feeds = wx.BoxSizer(wx.HORIZONTAL)
        feeds.Add(self.left_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        feeds.Add(self.right_bitmap, 1, wx.EXPAND | wx.ALL, 5)

        self.capture_button = wx.Button(self.panel, label="Capture Pair (0 saved)")
        self.capture_button.Bind(wx.EVT_BUTTON, self.on_capture)
        self.status = wx.StaticText(self.panel, label="Looking for chessboard...")

        controls = wx.BoxSizer(wx.HORIZONTAL)
        controls.Add(self.capture_button, 0, wx.ALL, 5)
        controls.Add(self.status, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(feeds, 1, wx.EXPAND | wx.ALL, 5)
        main.Add(controls, 0, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(main)

        OUTPUT_DIR.mkdir(exist_ok=True)
        self.pair_count = len(list(OUTPUT_DIR.glob("left_*.png")))
        self.capture_button.SetLabel(f"Capture Pair ({self.pair_count} saved)")

        # ------------------------- camera -------------------------
        # No width/height request here ON PURPOSE: calibration must be
        # captured at the same resolution the later examples run at,
        # which is the camera's default mode unless you change both.
        self.camera = StereoCamera(self.camera_index)
        if not self.camera.is_opened():
            self.status.SetLabel(
                f"WARNING: could not open camera {self.camera_index}")

        self.latest_pair = None       # raw full-res (left, right)
        self.both_detected = False
        self._busy = False

        self.capture_thread = StereoCaptureThread(self.camera)
        self.capture_thread.start()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.timer.Start(int(1000 / 15))

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
        # Raw full-resolution frames are what get SAVED; keep copies so
        # a capture click can't race the next thread update.
        self.latest_pair = (left.copy(), right.copy())

        # Detection + overlay run on a downscaled copy. Detection here
        # is only a gate ("is the whole board visible in both eyes?");
        # calibrate_stereo.py re-finds corners on the saved full-res
        # images at sub-pixel accuracy, so nothing is lost.
        small_l = scale_to_fit(left, self.detect_max_w, self.detect_max_h)
        small_r = scale_to_fit(right, self.detect_max_w, self.detect_max_h)
        gray_l = cv2.cvtColor(small_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(small_r, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, flags)

        display_l, display_r = small_l.copy(), small_r.copy()
        if found_l:
            cv2.drawChessboardCorners(display_l, CHESSBOARD, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(display_r, CHESSBOARD, corners_r, found_r)

        self.both_detected = found_l and found_r
        self.status.SetLabel("Board detected in BOTH eyes - OK to capture"
                             if self.both_detected
                             else "Looking for chessboard...")

        # Previews are display-only: pin to a fixed size so the layout
        # never reshuffles as frames arrive.
        prev_l = scale_to_fit(display_l, self.preview_w, self.preview_h)
        prev_r = scale_to_fit(display_r, self.preview_w, self.preview_h)
        self.left_bitmap.SetBitmap(cv_to_wx_bitmap(prev_l))
        self.right_bitmap.SetBitmap(cv_to_wx_bitmap(prev_r))

    def on_capture(self, event):
        if self.latest_pair is None or not self.both_detected:
            self.status.SetLabel("Not captured: board must be visible in both eyes")
            return
        left, right = self.latest_pair
        stamp = self.pair_count
        cv2.imwrite(str(OUTPUT_DIR / f"left_{stamp:02d}.png"), left)
        cv2.imwrite(str(OUTPUT_DIR / f"right_{stamp:02d}.png"), right)
        self.pair_count += 1
        self.capture_button.SetLabel(f"Capture Pair ({self.pair_count} saved)")

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
    frame = CaptureFrame(None, 'Example 2a: Capture Calibration Pairs', cam_idx)
    frame.Show()
    app.MainLoop()