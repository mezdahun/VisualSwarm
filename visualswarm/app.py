import logging
import os

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = logging.getLevelName(os.getenv('LOG_LEVEL', 'DEBUG'))
logger.setLevel(LOG_LEVEL)


def health():
    """Entrypoint to start high level application"""
    logger.info("VisualSwarm application OK!")
