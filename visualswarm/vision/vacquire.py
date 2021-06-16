"""
@author: mezdahun
@description: Acquiring low-level imput from camera module
"""
import logging

from visualswarm.contrib import simulation

if not simulation.ENABLE_SIMULATION:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    from picamera.exc import PiCameraValueError
else:
    import numpy as np   # pragma: simulation no cover

import cv2
import time
from datetime import datetime

from visualswarm.contrib import camera, logparams, monitoring

# if monitoring.ENABLE_CLOUD_LOGGING:
#     import google.cloud.logging
#     import os
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = monitoring.GOOGLE_APPLICATION_CREDENTIALS
#     # Instantiates a client
#     client = google.cloud.logging.Client()
#     client.get_default_handler()
#     client.setup_logging()

# using main logger
if not simulation.ENABLE_SIMULATION:
    # setup logging
    import os
    ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
    logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
    logger.setLevel(monitoring.LOG_LEVEL)
else:
    logger = logging.getLogger('visualswarm.app_simulation')   # pragma: simulation no cover
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

            # stabilize_color_space_params(picam)

            # Generates a 3D RGB array and stores it in rawCapture
            raw_capture = PiRGBArray(picam, size=camera.RESOLUTION)

            # Wait a certain number of seconds to allow the camera time to warmup
            logger.info('Waiting for camera warmup!')
            time.sleep(8)
            logger.info('--proceed--')
            frame_id = 0
            for frame in picam.capture_continuous(raw_capture,
                                                  format=camera.CAPTURE_FORMAT,
                                                  use_video_port=camera.USE_VIDEO_PORT):
                # Grab the raw NumPy array representing the image
                image = cv2.flip(frame.array, -1)

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
    except PiCameraValueError:   # pragma: no cover
        pass


def simulated_vision(raw_vision_stream):   # pragma: simulation no cover
    t = datetime.now()
    frame_id = 0
    while True:
        # enforcing checks on a regular basis
        if abs(t - datetime.now()).total_seconds() > (1 / camera.FRAMERATE):
            image = np.zeros((camera.RESOLUTION[0], camera.RESOLUTION[1], 3), np.uint8)
            image[10:15, 10:15, :] = 155
            raw_vision_stream.put((image, frame_id, t))
            t = datetime.now()
            frame_id += 1
