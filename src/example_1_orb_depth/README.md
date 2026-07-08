# Example 1: ORB Keypoints + Naive Stereo Depth

The original prototype, refactored for a sort-of tutorial series.
Detects ORB keypoints in the left lens of the camera, computes a StereoBM disparity map on the raw (unrectified) pair, and plots keypoints in 3D using `1/disparity` as a pseudo-depth.

Run:
```
python gui_frame.py [camera_index]
```

## What changed from the first prototype

* **StereoBM fixed-point output**: `StereoBM.compute()` returns disparity
  multiplied by 16 as `int16`. The value is now divided by 16 before use --
  previously the pseudo-depths were 16x too small.
* **Single-channel bitmap fix**: the disparity map is grayscale, but the old
  conversion assumed 3-channel BGR (`COLOR_BGR2RGB` on a 2D array raises an
  error). `common/conversions.py` handles both cases.
* Detector/matcher constructed once in `__init__`, not per frame; duplicate
  grayscale conversions removed; conversions moved to the GUI boundary so the
  vision code works in plain numpy.
* Scatter plot capped at 300 points per frame to keep redraws responsive.
* General UI implementation cleanup.

## Known limitation (on purpose...sort of)

Without rectification, StereoBM is matching along rows that are *not* true
epipolar lines, and `1/disparity` has no units. Depth here is ordinal at best.

This was a driving factor behind some of the issues seen in the first iteration of this project
and adjusting for (or attempting to automatically compensate for) this was causing modulairty issues.

That limitation is the motivation for Example 2 (and for having multiple examples).
