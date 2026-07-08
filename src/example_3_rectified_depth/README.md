# Example 3: Rectified Metric Depth (SGBM)

Loads the Example 2 calibration, rectifies each pair so epipolar lines are
horizontal rows, computes dense disparity with StereoSGBM, and reprojects to a
true 3D point cloud (`cv2.reprojectImageTo3D` with the `Q` matrix). 

The cloud is plotted in millimeters with real image colors (my favorite part!!)

Run:
```
python rectified_depth_gui.py [camera_index]
```

## Things to Try

* Toggle the epipolar guide lines and verify a feature visible in both lenses sits
  on the same green row. This is the rectification working!
* Hold an object at a measured distance (e.g. 500 mm) and check the cloud.
* SGBM parameters worth experimenting with: `numDisparities` (range of depths;
  must be divisible by 16 -- larger sees closer objects but costs time) and
  `blockSize` (smaller = more detail + more noise).

Falls back to an approximate camera model if no calibration file exists, with a
warning. The geomery is roughly right but the scale will not be.
