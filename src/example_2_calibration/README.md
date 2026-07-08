# Example 2: Stereo Calibration

Two scripts in this example:

1. **`capture_calibration_pairs.py [camera_index]`** -- live wx GUI that detects
   a 9x6-inner-corner chessboard in both lenses and saves synchronized pairs to
   `calibration_images/`. Captures 15-25 pairs: vary distance, tilt, and position
   (cover the image corners, where lens distortion is strongest). The button only
   saves when the board is found in BOTH lenses of the camera.

2. **`calibrate_stereo.py [--square-size MM]`** -- finds sub-pixel corners,
   calibrates each camera (intrinsics `K`, distortion `D`), runs
   `cv2.stereoCalibrate` for the relative pose (`R`, `T` -- the magnitude of `T`
   is the baseline), and `cv2.stereoRectify` for the rectification transforms and
   the `Q` reprojection matrix. Saves everything to `stereo_calibration.npz`.

`--square-size` is the printed edge length of one square in millimeters; it sets
the absolute scale of every later measurement. Measure it with a ruler after
printing (printers rescale!). (I also just use a real chessboard in these examples)


## Troubleshooting Tips

* Stereo RMS reprojection error under ~0.5 px is good; over ~1.0 px, recapture.
* The reported baseline should match a ruler measurement of the lens spacing
  (this camera is roughly 60 mm).
* A chessboard off of Amazon works fine too if that's already on hand
