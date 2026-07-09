# Example 3: Rectified Metric Depth (SGBM)

Loads the Example 2 calibration, rectifies each pair so epipolar lines are
horizontal rows, computes dense disparity with StereoSGBM, and reprojects to a
true 3D point cloud (`cv2.reprojectImageTo3D` with the `Q` matrix).
The cloud is plotted in millimeters with real image colors (my favorite part!!)

Run:
```
python rectified_depth_gui.py [camera_index]
```
The index argument is optional; `DEFAULT_CAMERA_INDEX` at the bottom of the
script is set to 1 (where the stereo rig enumerates on this machine).

## Things to Try

* Toggle the epipolar guide lines and verify a feature visible in both lenses sits
  on the same green row. This is the rectification working!
* Hold an object at a measured distance (e.g. 500 mm) and check the cloud.
* SGBM parameters worth experimenting with: `numDisparities` (range of depths;
  must be divisible by 16 -- larger sees closer objects but costs time) and
  `blockSize` (smaller = more detail + more noise).

## Implementation notes

* **The imports were broken in the first version of this file**: `sys.path`
  gets `src/` appended, so modules import as `common.*` -- the old
  `src.common.*` imports raised `ModuleNotFoundError` on launch.
* Same GUI hygiene as the other examples: camera reads on a background
  thread, a reentrancy guard + try/except around the timer handler
  (SGBM at 128 disparities is the heaviest per-frame load in the repo),
  and previews pinned to a fixed size so the layout never reshuffles.
* Previews use a local `fit_preview()` instead of
  `common.conversions.scale_to_fit` because they should also UPSCALE to
  fill their box; `scale_to_fit` is shrink-only on purpose. Change
  `preview_w/h` in the config block to resize them.
* The 3D cloud (`reprojectImageTo3D` + the matplotlib scatter) is the
  slowest stage after SGBM itself, so it only refreshes every
  `plot_every` frames; the video and disparity views update every frame.
* SGBM always runs at full rectified resolution -- the calibration's
  geometry is locked to the resolution it was captured at, and the
  status line warns if the calibration and camera sizes disagree.

Falls back to an approximate camera model if no calibration file exists, with a
warning. The geometry is roughly right but the scale will not be.