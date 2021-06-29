"""
@author: mezdahun
@description: Parameters related to grafana monitoring and ifdb configuration
"""
import os

# Parameters to save and visualize projection field with grafana and influxdb
SAVE_PROJECTION_FIELD = False
DOWNGRADING_FACTOR = 10

# Parameters regarding monitoring of flocking parameters of the main algorithm
SAVE_CONTROL_PARAMS = False

# Enabling cloud logging via Google Cloud
ENABLE_CLOUD_LOGGING = False
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/home/pi/VisualSwarm/GKEY.json')
