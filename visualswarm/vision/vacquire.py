"""
@author: mezdahun
@description: Acquiring raw imput from camera module
"""

from picamera import PiCamera
from picamera.array import PiRGBArray

import time
import cv2


def visual_input(process_queue):
    for i in range(100):
        process_queue.put(['this', 'is', 'now', i])
        time.sleep(0.5)

def visual_processor(process_queue):
    for j in range(30):
        print(process_queue.get())

def start_vision_stream():
    """Acquiring single image with picamera package"""
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32

    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(camera, size=(640, 480))

    # Wait a certain number of seconds to allow the camera time to warmup
    time.sleep(0.1)

    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        # Grab the raw NumPy array representing the image
        image = frame.array

        # Display the frame using OpenCV
        cv2.imshow("Frame", image)

        # Wait for keyPress for 1 millisecond
        key = cv2.waitKey(1) & 0xFF

        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
