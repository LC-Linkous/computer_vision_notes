# Example 2: Stereo Calibration

Two scripts in this example:

1. **`capture_calibration_pairs.py [camera_index]`** -- live wx GUI that detects
   a chessboard in both lenses and saves synchronized pairs to
   `calibration_images/`. Captures 15-25 pairs: vary distance, tilt, and position
   (cover the image corners, where lens distortion is strongest). The button only
   saves when the board is found in BOTH lenses of the camera.

   The index argument is optional; `DEFAULT_CAMERA_INDEX` at the bottom of the
   script is set to 1 (where the stereo rig enumerates on the test computer).

   Implementation notes: camera reads run on a background thread and live
   chessboard detection runs on a downscaled copy so the UI stays responsive
   (detection here is just a "board fully visible?" gate). **Saved images are always the raw full-resolution frames**; 
   they set the calibration's `image_size`, which must match the resolution the later examples run at.
   `calibrate_stereo.py` re-finds corners on the saved images at sub-pixel
   accuracy, so nothing is lost by detecting small in the live view.

2. **`calibrate_stereo.py [--square-size MM]`** -- finds sub-pixel corners,
   calibrates each camera (intrinsics `K`, distortion `D`), runs
   `cv2.stereoCalibrate` for the relative pose (`R`, `T` -- the magnitude of `T`
   is the baseline), and `cv2.stereoRectify` for the rectification transforms and
   the `Q` reprojection matrix. Saves everything to `stereo_calibration.npz`.

## The board

`CHESSBOARD` (defined at the top of BOTH scripts -- they must agree) is the
count of INNER corners, not squares:

* `(7, 7)` -- a real 8x8-square chessboard. **This is the current default.**
* `(9, 6)` -- the classic printed OpenCV pattern (10x7 squares).

Real-chessboard caveats:

* A 7x7 corner grid looks identical rotated 180 degrees, so OpenCV can order
  the corners from either end. Keep the board in the same orientation for
  every capture (or tape a marker on one corner square). If the stereo RMS
  comes out high despite sharp images, suspect this first.
* Glossy/varnished boards glare under overhead light -- tilt the board or
  move the light if detection flickers.
* Don't mix patterns: clear out `calibration_images/` before capturing with
  a different board.

`--square-size` is the edge length of one square in millimeters; it sets the
absolute scale of every later measurement. Printed pattern: measure with a
ruler after printing (printers rescale!). Real chessboard: squares are usually
50-57 mm -- measure yours, e.g. `python calibrate_stereo.py --square-size 55`.
Get this wrong and every depth in Examples 3-4 is scaled by the same ratio.

## Troubleshooting Tips

* Stereo RMS reprojection error under ~0.5 px is good; over ~1.0 px, recapture.
* The reported baseline should match a ruler measurement of the lens spacing
  (this camera is roughly 60 mm). If it's off by a suspiciously clean ratio,
  your `--square-size` is wrong.
* A chessboard off of Amazon works fine too if that's already on hand. 
Just set `CHESSBOARD` to match its inner-corner count.