"""
@author: mezdahun
@description: Acquiring low-level imput from camera module
"""
import logging
from picamera import PiCamera
from picamera.array import PiRGBArray

import time

from visualswarm.contrib import camera, logparams


# using main logger
logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors


def raw_vision(raw_vision_stream):
    """Process to capture raw input via the camera module and sequentially push it to a vision stream so that other
    processes can consume this stream
        Args:
            raw_vision_stream: multiprocessing.Queue type object to create stream for captured camera data.
        Returns:
            -shall not return-
    """
    picam = PiCamera()
    picam.resolution = camera.RESOLUTION
    picam.framerate = camera.FRAMERATE
    logger.debug(f'\n{bcolors.OKBLUE}--Camera Params--{bcolors.ENDC}\n'
                 f'{bcolors.OKBLUE}Resolution:{bcolors.ENDC} {camera.RESOLUTION} px\n'
                 f'{bcolors.OKBLUE}Frame Rate:{bcolors.ENDC} {camera.FRAMERATE} fps')

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
        raw_vision_stream.put(image)

        # Clear the raw capture stream in preparation for the next frame
        raw_capture.truncate(0)
