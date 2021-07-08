"""
@author: mezdahun
@description: Camera module related parameters
"""
import os
# Basic parameters
RESOLUTION = (320, 200)
FRAMERATE = 20
CAPTURE_FORMAT = "bgr"
USE_VIDEO_PORT = True
FLIP_CAMERA = bool(int(os.getenv('ENABLE_CLOUD_LOGGING', '1')))

# Stabilizing Color Space
FIX_ISO = False
ISO = 100
AWB_MODE = 'off'
