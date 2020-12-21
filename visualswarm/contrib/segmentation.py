"""
@author: mezdahun
@description: Parameters related to image segmentation
"""
from cv2 import cvtColor, COLOR_BGR2HSV
from numpy import uint8

# Interactive color tune
FIND_COLOR_INTERACTIVE = True

# Visualization on the fly
SHOW_VISION_STREAMS = False

# Change parameters here
TARGET_RGB_COLOR = (22, 55, 155)
HSV_HUE_RANGE = 43
SV_MINIMUM = 55
SV_MAXIMUM = 253

# Calculating secondary parameters (Do not change)
R, G, B = TARGET_RGB_COLOR
TARGET_HSV_COLOR = cvtColor(uint8([[[B, G, R]]]), COLOR_BGR2HSV)

HSV_LOW = uint8([TARGET_HSV_COLOR[0][0][0]-HSV_HUE_RANGE, SV_MINIMUM, SV_MINIMUM])
HSV_HIGH = uint8([TARGET_HSV_COLOR[0][0][0]+HSV_HUE_RANGE, SV_MAXIMUM, SV_MAXIMUM])
