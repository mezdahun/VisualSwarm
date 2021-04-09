"""
@author: mezdahun
@description: Parameters related to visualization
"""
from cv2 import cvtColor, COLOR_BGR2HSV
from numpy import uint8, pi

# Interactive color tune
FIND_COLOR_INTERACTIVE = True

# Visualization on the fly
SHOW_VISION_STREAMS = True
VIS_DOWNSAMPLE_FACTOR = 1

# Drawing, color in RGB
RAW_CONTOUR_COLOR = (0, 0, 255)
RAW_CONTOUR_WIDTH = 3
CONVEX_CONTOUR_COLOR = (0, 255, 0)
CONVEX_CONTOUR_WIDTH = 3

# Color Space Segmentation
NUM_SEGMENTATION_PROCS = 6

# Target color normal
# TARGET_RGB_COLOR = (207, 207, 0)
# Target color NoIR camera
TARGET_RGB_COLOR = (52, 74, 245)
HSV_HUE_RANGE = 29
SV_MINIMUM = 112
SV_MAXIMUM = 255
R, G, B = TARGET_RGB_COLOR
TARGET_HSV_COLOR = cvtColor(uint8([[[B, G, R]]]), COLOR_BGR2HSV)
HSV_LOW = uint8([TARGET_HSV_COLOR[0][0][0] - HSV_HUE_RANGE, SV_MINIMUM, SV_MINIMUM])
HSV_HIGH = uint8([TARGET_HSV_COLOR[0][0][0] + HSV_HUE_RANGE, SV_MAXIMUM, SV_MAXIMUM])

# VPF Preprocessing
GAUSSIAN_KERNEL_WIDTH = 15
MEDIAN_BLUR_WIDTH = 9
MIN_BLOB_AREA = 100

# Visual Projection
H_MARGIN = 10
W_MARGIN = 10
PHI_START = -0.5394 * pi
PHI_END = 0.5394 * pi
