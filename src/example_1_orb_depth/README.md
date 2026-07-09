# Example 1: ORB Keypoints + Naive Stereo Depth

The original prototype, refactored for a sort-of tutorial series.

Detects ORB keypoints in the left lens of the camera, computes a StereoBM
disparity map on the raw (unrectified) pair, and plots keypoints in 3D using
`1/disparity` as a pseudo-depth. The right-hand notebook has three tabs:

* **Point Cloud** -- 3D scatter of keypoints (x, y, pseudo-depth)
* **Heat Map** -- exponentially-decayed density of ORB keypoint hits, blended
  over the live left feed. Hot = plenty of trackable texture; cold = blank
  walls, glare, repeating patterns -- the places feature-based vision is
  blind. (Foreshadows Example 4: those hot zones are where the tracker will
  seed corners.)
* **Disparity Map** -- the normalized StereoBM output

Run:
```
python gui_frame.py [camera_index]
```
The index argument is optional; `DEFAULT_CAMERA_INDEX` at the bottom of
`gui_frame.py` is set to 1 (where the stereo rig enumerates on this machine).

## What changed from the first prototype

* **StereoBM fixed-point output**: `StereoBM.compute()` returns disparity
  multiplied by 16 as `int16`. The value is now divided by 16 before use --
  previously the pseudo-depths were 16x too small.
* **Single-channel bitmap fix**: the disparity map is grayscale, but the old
  conversion assumed 3-channel BGR (`COLOR_BGR2RGB` on a 2D array raises an
  error). `common/conversions.py` handles both cases.
* Detector constructed once in `__init__`, not per frame; duplicate grayscale
  conversions removed; conversions moved to the GUI boundary so the vision
  code works in plain numpy.
* Scatter plot capped at 300 points per frame to keep redraws responsive.
* General UI implementation cleanup.

## The "blank preview" bug (worth reading if you touch the GUI)

The first version of this refactor ran camera reads AND all the vision work
on the wx UI thread, on full-resolution frames, from a 24 fps `wx.Timer`.
Each tick took far longer than the timer interval, so timer events monopolized
the event loop and the *paint* events that actually draw bitmaps to screen
never ran. Symptom: the preview panels resized (that part happens
synchronously inside `SetBitmap`) but stayed black, and status text never
appeared. Fixes, all visible in `gui_frame.py`:

* Camera I/O lives in `common/capture_thread.py`; the timer handler just
  grabs the newest pair and never blocks on `VideoCapture.read()`.
* Frames are downscaled to 640x480 per eye (`proc_max_w/h`) before ORB and
  StereoBM run.
* Previews are display-only and pinned to a fixed size (`preview_w/h`), so
  the layout never reshuffles as frames arrive.
* A reentrancy guard skips a tick if the previous frame is still processing,
  and the matplotlib scatter (the slowest single draw call) only redraws
  every `plot_every` frames.
* The handler body is wrapped in try/except -- wx can swallow tracebacks, so
  errors are echoed to the status box and stderr instead of failing silently.

## Known limitation (on purpose...sort of)

Without rectification, StereoBM is matching along rows that are *not* true
epipolar lines, and `1/disparity` has no units. Depth here is ordinal at best.
This was a driving factor behind some of the issues seen in the first
iteration of this project, and adjusting for (or attempting to automatically
compensate for) this was causing modularity issues.

That limitation is the motivation for Example 2 (and for having multiple
examples).