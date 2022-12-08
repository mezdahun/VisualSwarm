"""
@author: mezdahun
@description: Parameters related to visualization
"""
from cv2 import cvtColor, COLOR_BGR2HSV
import numpy as np
import os

def calculate_reverse_mapping_fn(lens, orig_img_width):
    # calculate discretized reverse mapping for fisheye correction according to lens parameters
    lens['h_reverse_mapping'] = np.array(
        [np.round(num) for num in np.maximum(lens['a_nonlin'] * np.square(lens['h_domain_orig']),
                                             lens['a_lin'] * np.ones(orig_img_width) - lens['offset_lin'])])
    lens['new_width'] = np.sum(lens['h_reverse_mapping'])
    return lens


# Recognition Type, supported: 'Color' or 'CNN'
RECOGNITION_TYPE = "CNN"

# Interactive color tune
FIND_COLOR_INTERACTIVE = False

# Visualization on the fly
SHOW_VISION_STREAMS = bool(int(os.getenv('SHOW_VISION_STREAMS', '1')))
VIS_DOWNSAMPLE_FACTOR = 1

# Drawing, color in RGB
RAW_CONTOUR_COLOR = (0, 0, 255)
RAW_CONTOUR_WIDTH = 3
CONVEX_CONTOUR_COLOR = (0, 255, 0)
CONVEX_CONTOUR_WIDTH = 3

# Color Space Segmentation
if RECOGNITION_TYPE == "Color":
    NUM_SEGMENTATION_PROCS = 6
else:
    NUM_SEGMENTATION_PROCS = 1

# Target color normal
# TARGET_RGB_COLOR = (207, 207, 0)
# Target color NoIR camera
TARGET_RGB_COLOR = (52, 74, 245)
HSV_HUE_RANGE = 29
SV_MINIMUM = 112
SV_MAXIMUM = 255
R, G, B = TARGET_RGB_COLOR
TARGET_HSV_COLOR = cvtColor(np.uint8([[[B, G, R]]]), COLOR_BGR2HSV)
HSV_LOW = np.uint8([TARGET_HSV_COLOR[0][0][0] - HSV_HUE_RANGE, SV_MINIMUM, SV_MINIMUM])
HSV_HIGH = np.uint8([TARGET_HSV_COLOR[0][0][0] + HSV_HUE_RANGE, SV_MAXIMUM, SV_MAXIMUM])

# VPF Preprocessing
GAUSSIAN_KERNEL_WIDTH = 9  # 15
MEDIAN_BLUR_WIDTH = 5  # 9
MIN_BLOB_AREA = 0

# Visual Projection
FOV = float(os.getenv('ROBOT_FOV', '6.28'))
H_MARGIN = 1  # 10
W_MARGIN = 1  # 10
PHI_START = - (FOV / 2)  # * pi  # -0.5394 * pi
PHI_END = (FOV / 2)  # pi  # 0.5394 * pi

# CNN processing parameters
overlap_removal = bool(int(float(os.getenv('OVERLAP_REMOVAL', '0'))))
overlap_removal_thr = 0.5

# Take N largest projections
focus_on_N_largest = bool(int(float(os.getenv('FOCUS_N_NEAREST', '0'))))
N_largest = int(float(os.getenv('FOCUS_N_NEAREST', '4')))

# Taking blobs one-by-one
divided_projection_field = bool(int(float(os.getenv('MULTI_RETINA', '0'))))

# Fisheye lens approximate horizontal correction
# offsets: in pixel from left and right (with input resolution width 320px)
orig_img_width = 320  # we should get it from contrib.camera but can stay like this for now
# LENS1
lens1 = {
    'offset_left': 0,
    'offset_right': 25,
    'a_nonlin': 1 * np.pi,
    'a_lin': 5,
    'offset_lin': 0,
    'h_domain_orig': np.linspace(PHI_START, PHI_END, orig_img_width)
}
lens1 = calculate_reverse_mapping_fn(lens1, orig_img_width)

# LENS2
lens2 = {
    'offset_left': 7,
    'offset_right': 17,
    'a_nonlin': 1 * np.pi,
    'a_lin': 5,
    'offset_lin': 0,
    'h_domain_orig': np.linspace(PHI_START, PHI_END, orig_img_width)
}
lens2 = calculate_reverse_mapping_fn(lens2, orig_img_width)

# LENS3
lens3 = {
    'offset_left': 18,
    'offset_right': 3,
    'a_nonlin': 1 * np.pi,
    'a_lin': 5,
    'offset_lin': 0,
    'h_domain_orig': np.linspace(PHI_START, PHI_END, orig_img_width)
}
lens3 = calculate_reverse_mapping_fn(lens3, orig_img_width)

# LENS4
lens4 = {
    'offset_left': 13,
    'offset_right': 10,
    'a_nonlin': 1 * np.pi,
    'a_lin': 5,
    'offset_lin': 0,
    'h_domain_orig': np.linspace(PHI_START, PHI_END, orig_img_width)
}
lens4 = calculate_reverse_mapping_fn(lens4, orig_img_width)


# connect robots with lenses (labelled on HW elements with Ri and Li)
LENS_CONFIG={
    'Robot1': lens1,
    'Robot2': lens2,
    'Robot3': lens3,
    'Robot4': lens4
}

# use fisheye correction
USE_VPF_FISHEYE_CORRECTION = False
