"""
@author: mezdahun
@description: Acquiring low-level imput from camera module
"""

from picamera import PiCamera
from picamera.array import PiRGBArray

import time
import cv2


def visual_input(process_queue):
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
        process_queue.put(image)
        # cv2.imshow("Frame", image)

        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)

def visual_processor(process_queue):
    for j in range(35):
        img = process_queue.get()
        print(type(img))
        cv2.imshow("Frame", img)

def start_vision_stream():
    """Acquiring single image with picamera package"""
    pass
