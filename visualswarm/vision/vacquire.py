"""
@author: mezdahun
@description: Acquiring low-level imput from camera module
"""
import logging
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera.exc import PiCameraValueError

import time
from datetime import datetime

from visualswarm.contrib import camera, logparams

# using main logger
logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors


def stabilize_color_space_params(picam):
    """Method to fiy camera parameters such that the color space is uniform across captured frames.
    More info at: https://picamera.readthedocs.io/en/release-1.12/recipes1.html
        Args:
            picam: PiCamera instance to configure
        Returns:
            None
    """
    picam.iso = 300
    # Wait for the automatic gain control to settle
    time.sleep(2)
    # Now fix the values
    picam.shutter_speed = picam.exposure_speed
    picam.exposure_mode = 'off'
    g = picam.awb_gains
    picam.awb_mode = 'off'
    picam.awb_gains = g


def raw_vision(raw_vision_stream):
    """Process to capture raw input via the camera module and sequentially push it to a vision stream so that other
    processes can consume this stream
        Args:
            raw_vision_stream (multiprocessing.Queue): Stream to create stream for captured camera data.
        Returns:
            -shall not return-
    """
    try:
        try:
            picam = PiCamera()
            picam.resolution = camera.RESOLUTION
            picam.framerate = camera.FRAMERATE
            logger.debug(f'\n{bcolors.OKBLUE}--Camera Params--{bcolors.ENDC}\n'
                         f'{bcolors.OKBLUE}Resolution:{bcolors.ENDC} {camera.RESOLUTION} px\n'
                         f'{bcolors.OKBLUE}Frame Rate:{bcolors.ENDC} {camera.FRAMERATE} fps')

            stabilize_color_space_params(picam)

            # Generates a 3D RGB array and stores it in rawCapture
            raw_capture = PiRGBArray(picam, size=camera.RESOLUTION)

            # Wait a certain number of seconds to allow the camera time to warmup
            time.sleep(0.1)
            frame_id = 0
            for frame in picam.capture_continuous(raw_capture,
                                                  format=camera.CAPTURE_FORMAT,
                                                  use_video_port=camera.USE_VIDEO_PORT):
                # Grab the raw NumPy array representing the image
                image = frame.array

                # Adding time of capture for delay measurement
                capture_timestamp = datetime.utcnow()

                # pushing the captured image to the vision stream
                raw_vision_stream.put((image, frame_id, capture_timestamp))

                # Clear the raw capture stream in preparation for the next frame
                raw_capture.truncate(0)
                frame_id += 1
        except KeyboardInterrupt:
            try:
                pass
            except PiCameraValueError:
                pass
    except PiCameraValueError:
        pass
