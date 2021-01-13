"""
@author: mezdahun
@description: Parameters related to visual projection
"""
from numpy import pi

H_MARGIN = 60
W_MARGIN = 50

# Parameters to save and visualize projection field with grafana and influxdb
SAVE_PROJECTION_FIELD = True
DOWNGRADING_FACTOR = 5

# Visual field of the camera
PHI_START = -pi
PHI_END = pi