import time
import cv2
import wx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

from point_cloud import PointCloud

class GFrame(wx.Frame):
    def __init__(self, parent, title):
        super(GFrame, self).__init__(parent, title=title, size=(1300, 575))

        # Set default vars
        self.frame_rate = int(1000/24) #adjust as needed, must be an int
        self.video_idx = 1  # Change the index if needed
        self.showORB = True # Change to false to not show what is being detected
        self.point_cloud = PointCloud()

        # Create main panel for the frame elements
        self.panel = wx.Panel(self)

        # Create the top sizer and its elements
        # On the left are the stereoscopic camera feeds, and a textbox
        # On the right is a multi-page notebook displaying 3D graph data
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Create the camera panel and elemnts
        self.camera_panel = wx.Panel(self.panel)        
        self.left_panel = wx.Panel(self.camera_panel)
        self.right_panel = wx.Panel(self.camera_panel)

        # Create black default bitmap for camera panels
        default_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        default_bitmap = wx.Bitmap.FromBuffer(320, 240, default_frame)
        self.left_bitmap = wx.StaticBitmap(self.left_panel, bitmap=default_bitmap)
        self.right_bitmap = wx.StaticBitmap(self.right_panel, bitmap=default_bitmap)

        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_sizer.Add(self.left_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        self.right_sizer.Add(self.right_bitmap, 1, wx.EXPAND | wx.ALL, 5)

        self.left_panel.SetSizer(self.left_sizer)
        self.right_panel.SetSizer(self.right_sizer)

        top_camera_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_camera_sizer.Add(self.left_panel, 1, wx.EXPAND | wx.ALL, 5)
        top_camera_sizer.Add(self.right_panel, 1, wx.EXPAND | wx.ALL, 5)

        bottom_textbox_sizer = wx.BoxSizer(wx.VERTICAL)
        self.status_box = wx.TextCtrl(self.camera_panel, value="", style=wx.TE_MULTILINE|wx.TE_READONLY)
        bottom_textbox_sizer.Add(self.status_box, 1, wx.EXPAND | wx.ALL, 5)

        camera_sizer = wx.BoxSizer(wx.VERTICAL)
        camera_sizer.Add(top_camera_sizer, 2, wx.EXPAND | wx.ALL, 5)
        camera_sizer.Add(bottom_textbox_sizer, 1, wx.EXPAND | wx.ALL, 5)

        self.camera_panel.SetSizer(camera_sizer)

        # Create data notebook for right side of screen
        self.data_notebook = wx.Notebook(self.panel)

        #create panels and add to notebook
        # Set up heat map page (empty)
        self.heatmap_page = wx.Panel(self.data_notebook)
        heatmap_sizer = wx.BoxSizer(wx.VERTICAL)
        self.heatmap_page.SetSizer(heatmap_sizer)

        # Set up disparity map page (empty)
        self.disparity_page = wx.Panel(self.data_notebook)
        disparity_sizer = wx.BoxSizer(wx.VERTICAL)
        default_disparity_bitmap = wx.Bitmap.FromBuffer(320, 240, np.zeros((240, 320, 3), dtype=np.uint8))
        # Create a StaticBitmap widget for the disparity map
        self.disparity_bitmap = wx.StaticBitmap(self.disparity_page, bitmap=default_disparity_bitmap)
        # Add the StaticBitmap widget to the sizer
        disparity_sizer.Add(self.disparity_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        self.disparity_page.SetSizer(disparity_sizer)


        # Set up point cloud page (has matplot 3d plot)
        self.pointcloud_panel = wx.Panel(self.data_notebook)

        # Create matplotlib figure
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.pointcloud_panel, -1, self.fig)

        pointcloud_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pointcloud_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.pointcloud_panel.SetSizer(pointcloud_sizer)
        # Add pages to notebook
        self.data_notebook.AddPage(self.pointcloud_panel, "Point Cloud")
        self.data_notebook.AddPage(self.heatmap_page, "Heat Map")
        self.data_notebook.AddPage(self.disparity_page, "Disparity Map")

        # Add everything to sizers
        top_right_sizer = wx.BoxSizer(wx.VERTICAL)
        top_right_sizer.Add(self.data_notebook, 1, wx.EXPAND | wx.ALL, 5)

        top_sizer.Add(self.camera_panel, 1, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(top_right_sizer, 1, wx.EXPAND | wx.ALL, 5)


        # Create the bottom sizer elements
        self.buttons_panel = wx.Panel(self.panel)

        self.start_button = wx.Button(self.buttons_panel, label="Start Video")
        self.stop_button = wx.Button(self.buttons_panel, label="Stop Video")
        self.start_button.Bind(wx.EVT_BUTTON, self.start_video)
        self.stop_button.Bind(wx.EVT_BUTTON, self.stop_video)

        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        buttons_sizer.Add(self.start_button, 0, wx.ALL, 5)
        buttons_sizer.Add(self.stop_button, 0, wx.ALL, 5)

        self.buttons_panel.SetSizer(buttons_sizer)

        # Set up main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(top_sizer, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.buttons_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(main_sizer)

        self.capture = cv2.VideoCapture(self.video_idx)
        self.is_playing = False

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)

        self.Bind(wx.EVT_CLOSE, self.on_close)

    def update(self, event):
        ret, frame = self.capture.read()

        if ret:
            height, width, _ = frame.shape
            half_width = width // 2

            left_frame = frame[:, :half_width, :]
            right_frame = frame[:, half_width:, :]

            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

            left_bitmap = wx.Bitmap.FromBuffer(width // 2, height, left_frame)
            right_bitmap = wx.Bitmap.FromBuffer(width // 2, height, right_frame)


            left_bitmap, right_bitmap, disparity_bitmap, keypoint_coordinates = self.point_cloud.detectKeyPoints(left_bitmap, right_bitmap)

            self.left_bitmap.SetBitmap(left_bitmap)
            self.right_bitmap.SetBitmap(right_bitmap)
            self.disparity_bitmap.SetBitmap(disparity_bitmap)

            self.graphPointCloud(keypoint_coordinates)


        if not self.is_playing:
            self.timer.Stop()

    def start_video(self, event):
        self.is_playing = True
        self.timer.Start(self.frame_rate) 
        self.updateStatusText("starting video")

    def stop_video(self, event):
        self.is_playing = False
        self.updateStatusText("stopping video")

    def on_close(self, event):
        self.timer.Stop()
        self.capture.release()
        self.Destroy()
        wx.Exit()

    def updateStatusText(self, t):
        if t is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(t)  + "\n" 
        self.status_box.AppendText(msg)

    def graphPointCloud(self, keypoint_coordinates):
        keypoint_coordinates = np.array(keypoint_coordinates)
        try:
            self.ax.clear()
            self.ax.set_zlim(0, 0.5)  # Set the static z-axis range
            self.ax.scatter(keypoint_coordinates[:,0], keypoint_coordinates[:,1], keypoint_coordinates[:,2], c=keypoint_coordinates[:,2], cmap='plasma')
            #self.ax.plot_trisurf(keypoint_coordinates[:,0], keypoint_coordinates[:,1], keypoint_coordinates[:,2])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Estimated Depth (m)')
            self.canvas.draw()
        except:
            print("array doesnt have enough values for graphing")



if __name__ == '__main__':
    app = wx.App()
    frame = GFrame(None, 'Stereo Camera Viewer')
    frame.Show()
    app.MainLoop()
