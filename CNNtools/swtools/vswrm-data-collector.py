"""

    VSWRM Data Collector to train Single-Shot Detector CNNs

    The SW collects images in JPG format into a given directory (savedir).
    The framerate (framerate) and resolution of the camera (resolution) can
    be changed via flags. After starting the software hit Space to capture image or
    Esc to quit.

"""


import os
import argparse
import cv2
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera.exc import PiCameraValueError
from threading import Thread
import logging

from visualswarm.contrib import logparams
bcolors = logparams.BColors

logging.basicConfig()
logger = logging.getLogger('VSWRM-dataCollector')
logger.setLevel('INFO')



HELP_MESSAGE = """

    VSWRM Data Collector to train Single-Shot Detector CNNs
    
    The SW collects images in JPG format into a given directory (savedir). 
    The framerate (framerate) and resolution of the camera (resolution) can
    be changed via flags. After starting the software hit Space to capture image or
    Esc to quit.

"""


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-v', '--version', action='version',
                    version='v1.0', help="Show program's version number and exit.")
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help=HELP_MESSAGE)
parser.add_argument('-s', '--savedir', help='Folder the .jpg images are being saved. Default is current directory',
                    required=True, default=os.getcwd())
parser.add_argument('-r', '--resolution',
                    help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('-f', '--framerate', help='Framerate of the camera in [fps] as integer. Default is 30.',
                    default=30)

args = parser.parse_args()

SAVE_FOLDER = args.savedir
os.makedirs(SAVE_FOLDER, exist_ok=True)

RESOLUTION = [int(i) for i in args.resolution.split('x')]
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

FRAMERATE = int(args.framerate)

# try:
#     try:
# picam = PiCamera()
# picam.resolution = RESOLUTION
# picam.framerate = FRAMERATE
logger.info(f'\n\t{bcolors.OKBLUE}--Camera Params--{bcolors.ENDC}\n'
            f'\t{bcolors.OKBLUE}Resolution:{bcolors.ENDC} {RESOLUTION} px\n'
            f'\t{bcolors.OKBLUE}Frame Rate:{bcolors.ENDC} {FRAMERATE} fps\n'
            f'\n\t-- images are saved in {SAVE_FOLDER}\n'
            f'\t-- press {bcolors.OKBLUE}Space{bcolors.ENDC} to save frame.\n'
            f'\t-- press {bcolors.FAIL}Esc{bcolors.ENDC} or {bcolors.FAIL}Ctrl+C{bcolors.ENDC} to quit.\n')

# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(picam, size=RESOLUTION)

# Wait a certain number of seconds to allow the camera time to warmup
logger.info('--Waiting 8 seconds for PI-camera to warmup!')
time.sleep(8)
logger.info('--Start Video Stream')

cv2.namedWindow('Camera Stream', cv2.WINDOW_NORMAL)

# cam = cv2.VideoCapture(0)
videostream = VideoStream(resolution=(imW, imH), framerate=FRAMERATE).start()
time.sleep(1)

frame_id = 0
# for frame in picam.capture_continuous(raw_capture,
#                                       format='bgr',
#                                       use_video_port=True):
while True:
    ret, image = videostream.read()
    # Grab the raw NumPy array representing the image
    # image = frame.array

    cv2.imshow('Camera Stream', image)
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        logger.info("Escape was hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = os.path.join(SAVE_FOLDER, f"opencv_frame_{frame_id}.jpg")
        cv2.imwrite(img_name, image)
        logger.info(f"{img_name} saved!")

    # raw_capture.truncate(0)
    frame_id += 1

f'-- {bcolors.OKBLUE}Bye Bye!{bcolors.ENDC}'
#     except KeyboardInterrupt:
#         try:
#             f'-- {bcolors.FAIL}KeyboardInterrupt!{bcolors.ENDC} Exiting gracefully...'
#             f'-- {bcolors.OKBLUE}Bye Bye!{bcolors.ENDC}'
#             pass
#         except PiCameraValueError:
#             pass
# except PiCameraValueError:
#     logger.warning(f'-- {bcolors.FAIL}PiCameraError detected!{bcolors.ENDC}')
#     pass
