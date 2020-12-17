"""
@author: mezdahun
@description: Read env variables and store global project parameters
"""

import logging
import os

LOG_LEVEL = logging.getLevelName(os.getenv('LOG_LEVEL', 'DEBUG'))
