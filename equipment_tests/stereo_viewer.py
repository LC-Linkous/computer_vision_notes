##--------------------------------------------------------------------\
#   Camera Test
#   './src/GFrame.py'
#   Class for GUI layout and basic functionality
#
#   Author(s): Lauren Linkous (LINKOUSLC@vcu.edu)
#   Last update: April 2024
##--------------------------------------------------------------------\
import cv2
import wx

class StereoCameraViewer(wx.Frame):
    def __init__(self, parent, title):
        super(StereoCameraViewer, self).__init__(parent, title=title, size=(800, 500))

        self.panel = wx.Panel(self)

        self.camera_panel = wx.Panel(self.panel)
        self.buttons_panel = wx.Panel(self.panel)

        self.left_panel = wx.Panel(self.camera_panel)
        self.right_panel = wx.Panel(self.camera_panel)

        self.left_bitmap = wx.StaticBitmap(self.left_panel)
        self.right_bitmap = wx.StaticBitmap(self.right_panel)

        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_sizer.Add(self.left_bitmap, 1, wx.EXPAND | wx.ALL, 5)
        self.right_sizer.Add(self.right_bitmap, 1, wx.EXPAND | wx.ALL, 5)

        self.left_panel.SetSizer(self.left_sizer)
        self.right_panel.SetSizer(self.right_sizer)

        camera_sizer = wx.BoxSizer(wx.HORIZONTAL)
        camera_sizer.Add(self.left_panel, 1, wx.EXPAND | wx.ALL, 5)
        camera_sizer.Add(self.right_panel, 1, wx.EXPAND | wx.ALL, 5)

        self.camera_panel.SetSizer(camera_sizer)

        self.start_button = wx.Button(self.buttons_panel, label="Start Video")
        self.stop_button = wx.Button(self.buttons_panel, label="Stop Video")
        self.start_button.Bind(wx.EVT_BUTTON, self.start_video)
        self.stop_button.Bind(wx.EVT_BUTTON, self.stop_video)

        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        buttons_sizer.Add(self.start_button, 0, wx.ALL, 5)
        buttons_sizer.Add(self.stop_button, 0, wx.ALL, 5)

        self.buttons_panel.SetSizer(buttons_sizer)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.camera_panel, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.buttons_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(main_sizer)

        self.capture = cv2.VideoCapture(1)  # Change the index if needed
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

            self.left_bitmap.SetBitmap(left_bitmap)
            self.right_bitmap.SetBitmap(right_bitmap)

        if not self.is_playing:
            self.timer.Stop()

    def start_video(self, event):
        self.is_playing = True
        self.timer.Start(int(1000/24))  # Adjust the frame rate as needed

    def stop_video(self, event):
        self.is_playing = False

    def on_close(self, event):
        self.timer.Stop()
        self.capture.release()
        self.Destroy()

if __name__ == '__main__':
    app = wx.App()
    frame = StereoCameraViewer(None, 'Stereo Camera Viewer')
    frame.Show()
    app.MainLoop()
