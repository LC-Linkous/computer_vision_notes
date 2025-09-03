import numpy as np
import cv2
import wx

class PointCloud:
    def __init__(self):
        # Initialize any variables or parameters needed for point cloud processing
        self.currentPoints = []
        self.orb = cv2.ORB_create()

    def detectKeyPoints(self, left_bitmap, right_bitmap):
        # This function takes in the current video feed as 2 bitmaps (one for the left lens, and one for the right),
        # and using the data it finds common markers that are used to estimate the depth of a point of interest from the lens
        # Convert wx.Bitmap to numpy array
            image1 = self.wx_bitmap_to_cv(left_bitmap)
            image2 = self.wx_bitmap_to_cv(right_bitmap)
            
            # Convert images to grayscale
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Initialize the ORB detector
            orb = cv2.ORB_create()
                    
            # Detect keypoints and compute descriptors for both images
            keypoints1, descriptors1 = orb.detectAndCompute(image1_gray, None)
            keypoints2, descriptors2 = orb.detectAndCompute(image2_gray, None)
            
            # Draw keypoints on the images
            image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
            image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None)
            
            # Convert images to grayscale
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity map
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
            disparity = stereo.compute(image1_gray, image2_gray)
            
            # Normalize disparity map for better visualization
            disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            depths, keypoint_coordinates = self.relateKeypointsToDepth(keypoints1,disparity)
            
            # Convert images back to wx.Bitmap
            image1_bitmap_with_keypoints = self.cv_to_wx_bitmap(image1_with_keypoints)
            image2_bitmap_with_keypoints = self.cv_to_wx_bitmap(image2_with_keypoints)
            disparity_bitmap = self.cv_to_wx_bitmap(disparity_norm)
            
            return image1_bitmap_with_keypoints, image2_bitmap_with_keypoints, disparity_bitmap, keypoint_coordinates #keypoints1, depths # keypoints2, disparity


    def wx_bitmap_to_cv(self, bitmap):
        width, height = bitmap.GetSize()
        buffer = bytearray(width * height * 3)
        bitmap.CopyToBuffer(buffer)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
        return img

    def cv_to_wx_bitmap(self, image):
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        wx_image = wx.Image(width, height, image)
        return wx.Bitmap(wx_image)
    

    def relateKeypointsToDepth(self,keypoints1,disparity):
        # Associate keypoints with depth
        depths = []
        keypoint_coordinates = []
        for kp in keypoints1:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            disparity_value = disparity[y, x]
            if disparity_value != 0:
                depth = max(0, 1.0 / disparity_value)  # Inverse depth, but making sure not negative distance
                depth = depth *100 # convert to cm
                depths.append(depth)
                keypoint_coordinates.append([kp.pt[0], kp.pt[1], depth])
            else:
                depths.append(0.0)  # If no disparity, set depth to 0

        return depths, keypoint_coordinates


    def getPoints(self):
        # Return the 3D data points for plotting
        # This function will return the calculated 3D points for the point cloud
        return self.keyPoints

    def bitmapToFrame(self, bitmap):
        # Convert wx.Bitmap to numpy array (OpenCV frame)
        w, h = bitmap.GetWidth(), bitmap.GetHeight()
        bmp_str = bitmap.ConvertToImage().GetData()
        frame = np.frombuffer(bmp_str, dtype=np.uint8)
        frame = frame.reshape((h, w, 3))

        return frame
