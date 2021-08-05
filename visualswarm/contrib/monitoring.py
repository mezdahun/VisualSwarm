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
SAVE_CONTROL_PARAMS = True

# Enabling cloud logging via Google Cloud
ENABLE_CLOUD_LOGGING = bool(int(os.getenv('ENABLE_CLOUD_LOGGING', '0')))
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/home/pi/VisualSwarm/GKEY.json')
LOG_LEVEL = logging.getLevelName(os.getenv('LOG_LEVEL', 'DEBUG'))

# Saving visual stream as video if requested
SAVE_VISION_VIDEO = bool(int(os.getenv('SAVE_VISION_VIDEO', '0')))
SAVED_VIDEO_FOLDER = '/home/pi/VisualSwarm/videos'
DRIVE_SHARED_FOLDER_ID = "1M3_D-bh5r9wFRQh7KIwoqUCPLXjljjVF"

# Collect training data during experiment to finetune CNN based vision
SAVE_CNN_TRAINING_DATA = bool(int(os.getenv('SAVE_CNN_TRAINING_DATA', '0')))
CNN_TRAINING_DATA_FREQ = 1  # in Hz

# Uploading saved videos to Google Drive
ENABLE_CLOUD_STORAGE = bool(int(os.getenv('ENABLE_CLOUD_STORAGE', '1')))
CLOUD_STORAGE_AUTH_MODE = 'ServiceAccount'  # 'ServiceAccount' or 'OAuth2'
