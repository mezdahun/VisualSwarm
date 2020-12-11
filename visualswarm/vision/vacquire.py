"""
@author: mezdahun
@description: Acquiring raw imput from camera module
"""

from picamera import PiCamera
from time import sleep


def acq_image():
    """Acquiring single image with picamera package"""
    camera = PiCamera()
    camera.start_preview()
    sleep(5)
    camera.stop_preview()
