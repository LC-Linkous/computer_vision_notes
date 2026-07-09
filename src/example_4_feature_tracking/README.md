# Example 4: Temporal Feature Tracking + Depth

Examples 1-3 match features *across the stereo pair* (one instant in time).
This example matches *across time*: Shi-Tomasi corners are followed frame-to-frame
with pyramidal Lucas-Kanade optical flow (with a forward-backward consistency
check to drop bad tracks), and each live track samples the SGBM disparity map to
get a metric depth. Tracks are drawn as trails colored by depth, plus a top-down
(X-Z) scatter of where tracked features sit in space.

Run:
```
python tracking_gui.py [camera_index]
```

## Why this matters for the roadmap

A depth map tells you where surfaces are *now*; tracks tell you how points move
*between* frames. Visual odometry (Example 5) and SLAM both live on exactly this
data association problem. The forward-backward check here is a small taste of the
outlier rejection that dominates real VO/SLAM pipelines.
