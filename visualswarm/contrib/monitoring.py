"""
@author: mezdahun
@description: Parameters related to grafana monitoring and ifdb configuration
"""
import os
import logging

# Parameters to save and visualize projection field with grafana and influxdb
SAVE_PROJECTION_FIELD = False
DOWNGRADING_FACTOR = 10

# Parameters regarding monitoring of flocking parameters of the main algorithm
SAVE_CONTROL_PARAMS = False

# Enabling cloud logging via Google Cloud
ENABLE_CLOUD_LOGGING = bool(int(os.getenv('ENABLE_CLOUD_LOGGING', '0')))
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/home/pi/VisualSwarm/GKEY.json')
LOG_LEVEL = logging.getLevelName(os.getenv('LOG_LEVEL', 'DEBUG'))

# Saving visual stream as video if requested
SAVE_VISION_VIDEO = bool(int(os.getenv('SAVE_VISION_VIDEO', '0')))
SAVED_VIDEO_FOLDER = '/home/pi/VisualSwarm/videos'