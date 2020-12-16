"""
@author: mezdahun
@description: Acquiring low-level imput from camera module
"""

from picamera import PiCamera
from picamera.array import PiRGBArray

import time
import cv2

from visualswarm.contrib import camera


def visual_input(vision_stream):
    """Process to capture raw input via the camera module and sequentially push it to a vision stream so that other
    processes can consume this stream
        Args:
            vision_stream: multiprocessing.Queue type object to create stream for captured camera data.
        Returns:
            -shall not return-"""
    picam = PiCamera()
    picam.resolution = camera.RESOLUTION
    picam.framerate = camera.FRAMERATE

    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(picam, size=camera.RESOLUTION)

    # Wait a certain number of seconds to allow the camera time to warmup
    time.sleep(0.1)

    for frame in picam.capture_continuous(raw_capture,
                                          format=camera.CAPTURE_FORMAT,
                                          use_video_port=camera.USE_VIDEO_PORT):
        # Grab the raw NumPy array representing the image
        image = frame.array

        # pushing the captured image to the vision stream
        vision_stream.put(image)

        # Clear the raw capture stream in preparation for the next frame
        raw_capture.truncate(0)


def visual_processor(process_queue):
    for j in range(2000):
        img = process_queue.get()
        print(type(img))
        cv2.imshow("Frame", img)
        cv2.waitKey(1)


def start_vision_stream():
    """Acquiring single image with picamera package"""
    pass
