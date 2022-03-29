"""
@author: mezdahun
@description: Camera module related parameters
"""
import os
# Basic parameters
RESOLUTION = (int(os.getenv('RES_WIDTH', '320')), int(os.getenv('RES_HEIGHT', '200')))
print(RESOLUTION)
FRAMERATE = 20
CAPTURE_FORMAT = "bgr"
USE_VIDEO_PORT = True
FLIP_CAMERA = bool(int(os.getenv('FLIP_CAMERA', '1')))

# Stabilizing Color Space
FIX_ISO = False
ISO = 100
AWB_MODE = 'off'
