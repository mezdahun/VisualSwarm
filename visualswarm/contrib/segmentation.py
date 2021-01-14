"""
@author: mezdahun
@description: Parameters related to image segmentation
"""
from cv2 import cvtColor, COLOR_BGR2HSV
from numpy import uint8

# Number of segmentation processes
NUM_SEGMENTATION_PROCS = 6

# Change parameters here
# To interactively tune these set FIND_COLOR_INTERACTIVE to True
TARGET_RGB_COLOR = (207, 207, 0)
HSV_HUE_RANGE = 14
SV_MINIMUM = 78
SV_MAXIMUM = 255

# Blur
GAUSSIAN_KERNEL_WIDTH = 15
MEDIAN_BLUR_WIDTH = 9

# Others
MIN_BLOB_AREA = 100

# Calculating secondary parameters (Do not change)
R, G, B = TARGET_RGB_COLOR
TARGET_HSV_COLOR = cvtColor(uint8([[[B, G, R]]]), COLOR_BGR2HSV)

HSV_LOW = uint8([TARGET_HSV_COLOR[0][0][0]-HSV_HUE_RANGE, SV_MINIMUM, SV_MINIMUM])
HSV_HIGH = uint8([TARGET_HSV_COLOR[0][0][0]+HSV_HUE_RANGE, SV_MAXIMUM, SV_MAXIMUM])
