"""
@author: mezdahun
@description: Read env variables and store global project parameters
"""

import logging
import os

LOG_LEVEL = logging.getLevelName(os.getenv('LOG_LEVEL', 'DEBUG'))

# influx configuration from .env.local or default
INFLUX_FRESH_DB_UPON_START = os.getenv('INFLUX_FRESH_DB_START', False)
INFLUX_USER = os.getenv('INFLUX_USER', 'grafana')
INFLUX_PSWD = os.getenv('INFLUX_PSWD')
INFLUX_DB_NAME = os.getenv('INFLUX_DB_NAME', 'home')
INFLUX_HOST = os.getenv('INFLUX_HOST', '127.0.0.1')
INFLUX_PORT = os.getenv('INFLUX_HOST', '8086')

# testing parameters
EXIT_CONDITION = False

