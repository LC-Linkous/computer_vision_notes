# Example 5: Stereo Visual Odometry

This example is very buggy and needs some work for capture, processing, and display. 
It's likely the tipping point of needing to split the examples into multiple files and
plan a more solid based to continue work on rather than using stand-alone, single files.


Recovers the camera's own motion. Per frame:

1. Rectify; detect ORB in both eyes; match left-right along epipolar rows
   (Lowe ratio test + row check) and triangulate a metric 3D landmark per
   feature: `Z = fx * B / d`.
2. Match current left descriptors against the previous frame's.
3. Previous 3D landmarks + current 2D observations -> `cv2.solvePnPRansac`
   gives the relative camera motion.
4. Accumulate relative motions into a global pose; plot the trajectory top-down.

Run (a real calibration from Example 2 is strongly recommended):
```
python stereo_vo_gui.py [camera_index]
```
The index argument is optional; `DEFAULT_CAMERA_INDEX` at the bottom of the
script is set to 1 (where the stereo rig enumerates on this machine).

## Things to try

* Walk a slow loop around a textured room and compare start/end points on the
  plot. The gap is **drift** -- frame-to-frame errors compound because nothing
  ever corrects them.
* Point the camera at a blank wall and watch the inlier count collapse: VO
  needs texture.

## Implementation notes

* **The imports were broken in the first version of this file**: `sys.path`
  gets `src/` appended, so modules import as `common.*` -- the old
  `src.common.*` imports raised `ModuleNotFoundError` on launch.
* Same GUI hygiene as the other examples: camera reads on a background
  thread, a reentrancy guard + try/except around the timer handler (two
  ORB detections at 1500 features, two knnMatch passes, and PnP RANSAC
  per frame is the busiest pipeline in the series), a preview pinned to
  a fixed size, and warnings for a missing camera or a calibration whose
  resolution doesn't match the camera's.
* The trajectory plot redraws every `plot_every` frames, and paths longer
  than `max_plot_poses` are decimated FOR DRAWING ONLY (the final pose is
  always kept exact) -- the full trajectory is retained for the pose and
  path-length math. Without this, redraw cost grows without bound the
  longer you walk.
* Reset Trajectory also clears the previous frame's landmarks, so the
  first motion after a reset is estimated fresh instead of against
  pre-reset geometry.

## Relationship to SLAM

This is a SLAM *front-end*. 

Full SLAM adds a back-end: keyframes, a persistent
map, loop-closure detection (recognizing a previously seen place), and global
optimization (pose graphs / bundle adjustment) that snaps the drifted trajectory
back into consistency. 

This example's drift is the motivating problem for the
next example in this repo (and the next phase of the larger project)
This is the big step that mirrors the 2020 course project (hallway
loop reconstruction), but now with live hardware instead of provided video.