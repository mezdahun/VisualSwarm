"""
@author: mezdahun
@description: Parameters related to image segmentation
"""
from cv2 import cvtColor, COLOR_BGR2HSV
from numpy import uint8

# Change parameters here
TARGET_RGB_COLOR = (25, 50, 85)
HSV_HUE_RANGE = 15
SV_MINIMUM = 75
SV_MAXIMUM = 255

# Calculating secondary parameters (Do not change)
R, B, G = TARGET_RGB_COLOR
print(R)
print(B)
print(G)
TARGET_HSV_COLOR = cvtColor(uint8([[[B, G, R]]]), COLOR_BGR2HSV)
print(TARGET_HSV_COLOR)

HSV_LOW = uint8([TARGET_HSV_COLOR[0][0][0]-HSV_HUE_RANGE, SV_MINIMUM, SV_MINIMUM])
HSV_HIGH = uint8([TARGET_HSV_COLOR[0][0][0]+HSV_HUE_RANGE, SV_MAXIMUM, SV_MAXIMUM])

# lower_blue = np.array([90, 45, 45])
# upper_blue = np.array([115, 255, 255])

print(HSV_LOW)
print(HSV_HIGH)
