"""
@author: mezdahun
@description: Main app of visualswarm
"""

import logging
from multiprocessing import Process, Queue
from visualswarm import env
from visualswarm.vision import vacquire

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)


def health():
    """Entrypoint to start high level application"""
    logger.info("VisualSwarm application OK!")
    q = Queue()
    vinput = Process(target=vacquire.visual_input, args=(q,))
    vprocess = Process(target=vacquire.visual_processor, args=(q,))
    vinput.start()
    vprocess.start()
    vinput.join()
    vprocess.join()
