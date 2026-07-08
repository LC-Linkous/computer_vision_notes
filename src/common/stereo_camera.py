##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/common/stereo_camera.py'
#   Wrapper for a side-by-side stereo USB camera (single video device,
#   left and right images packed into one wide frame).
##--------------------------------------------------------------------\

import cv2


class StereoCamera:
    """Opens a single USB device that delivers a side-by-side stereo frame
    and splits it into (left, right) views.

    The 4MP dual-lens camera used in this project enumerates as ONE device
    delivering 3840x1080 (two 1920x1080 images side by side). Lower
    resolutions like 1280x480 (two 640x480) are also available and are much
    friendlier for real-time processing.
    """

    def __init__(self, index=0, width=None, height=None):
        self.index = index
        self.capture = cv2.VideoCapture(index)
        if width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self):
        return self.capture.isOpened()

    def read(self):
        """Returns (ok, left_bgr, right_bgr). Frames are None when ok is False."""
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None, None
        half = frame.shape[1] // 2
        left = frame[:, :half]
        right = frame[:, half:]
        return True, left, right

    def frame_size(self):
        """(width, height) of ONE eye after splitting."""
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        self.capture.release()
