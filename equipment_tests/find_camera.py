import cv2

def print_camera_info():
    for i in range(10):  # Try up to 10 devices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index: {i}")
            print(f"Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
            print(f"Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
            print()
            cap.release()

if __name__ == "__main__":
    print("Available camera devices:")
    print_camera_info()
