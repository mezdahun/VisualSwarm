import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera.exc import PiCameraValueError
import logging

logger = logging.getLogger('VSWRM-dataCollector')


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution',
                    help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

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
            image = frame.array

            # Adding time of capture for delay measurement
            capture_timestamp = datetime.utcnow()

            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = os.path.join(SAVE_FOLDER, f"opencv_frame_{frame_id}.jpg")
                cv2.imwrite(img_name, image)
                print("{img_name} written!")
                img_counter += 1

            frame_id += 1
    except KeyboardInterrupt:
        try:
            pass
        except PiCameraValueError:
            pass
except PiCameraValueError:
    pass