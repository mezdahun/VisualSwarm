"""
@author: mezdahun
@description: Main app of visualswarm
"""

import logging

from visualswarm import env

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)


def health():
    """Entrypoint to start high level application"""
    logger.info("VisualSwarm application OK!")
