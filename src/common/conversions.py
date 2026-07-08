##--------------------------------------------------------------------\
#   computer_vision_notes
#   './src/common/conversions.py'
#   Conversion helpers between OpenCV (numpy) images and wx bitmaps.
#
#   Keeping these in one place avoids the subtle bugs that come from
#   re-implementing them per-file (channel order, grayscale handling).
##--------------------------------------------------------------------\

import cv2
import numpy as np
import wx


def cv_to_wx_bitmap(image):
    """Convert an OpenCV image (BGR or single-channel grayscale) to a wx.Bitmap.

    Handles both 3-channel BGR frames and 2D grayscale arrays (e.g. a
    normalized disparity map). wx expects tightly-packed RGB bytes.
    """
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return wx.Bitmap.FromBuffer(width, height, rgb)


def wx_bitmap_to_cv(bitmap):
    """Convert a wx.Bitmap to an OpenCV BGR numpy array."""
    width, height = bitmap.GetSize()
    buffer = bytearray(width * height * 3)
    bitmap.CopyToBuffer(buffer)  # buffer is RGB
    rgb = np.frombuffer(bytes(buffer), dtype=np.uint8).reshape((height, width, 3))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def scale_to_fit(image, max_width, max_height):
    """Downscale an image to fit in a box while preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image
