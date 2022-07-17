"""
@author: mezdahun
@description: Webcam observation camera parameters
"""
import os
# Basic parameters
RESOLUTION = (1000, 500)
FRAMERATE = 20
CAPTURE_FORMAT = "bgr"
USE_VIDEO_PORT = True
FLIP_CAMERA = bool(int(os.getenv('FLIP_CAMERA', '1')))

# Stabilizing Color Space
FIX_ISO = True
ISO = 100
AWB_MODE = 'off'
