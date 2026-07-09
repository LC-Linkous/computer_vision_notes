##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/common/capture_thread.py'
#   Background capture thread so camera I/O never blocks the wx UI.
#
#   Why: wx.Timer callbacks run on the UI thread. VideoCapture.read()
#   blocks until a frame arrives, so as per-frame processing gets heavier
#   (SGBM, feature matching, PnP) the GUI starts to stutter and buffered
#   frames pile up, adding latency. This thread continuously drains the
#   camera and keeps only the most recent stereo pair; the UI grabs the
#   latest pair whenever it is ready to draw.
##--------------------------------------------------------------------\

import threading
import time


class StereoCaptureThread(threading.Thread):
    """Continuously reads from a StereoCamera, retaining the newest pair."""

    def __init__(self, stereo_camera):
        super().__init__(daemon=True)
        self.camera = stereo_camera
        self._lock = threading.Lock()
        self._latest = None          # (left, right) most recent pair
        self._running = threading.Event()
        self._running.set()

    def run(self):
        while self._running.is_set():
            ok, left, right = self.camera.read()
            if not ok:
                # Don't busy-spin a core if the camera is unplugged,
                # failed to open, or is mid-reconnect.
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = (left, right)

    def get_latest(self):
        """Returns (left, right) or None if no frame has arrived yet.
        Always returns the most recent pair; never blocks."""
        with self._lock:
            return self._latest

    def stop(self):
        self._running.clear()