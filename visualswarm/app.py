"""
@author: mezdahun
@description: Main app of visualswarm
"""

import logging
from multiprocessing import Process, Queue
from visualswarm import env
from visualswarm.vision import vacquire, vprocess

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)


def health():
    """Entrypoint to start high level application"""
    logger.info("VisualSwarm application OK!")


def start_vision_stream():
    """Start the visual stream of the Pi"""
    raw_vision_stream = Queue()
    high_level_vision_stream = Queue()
    raw_vision = Process(target=vacquire.raw_vision, args=(raw_vision_stream,))
    high_level_vision = Process(target=vprocess.high_level_vision, args=(raw_vision_stream, high_level_vision_stream,))
    try:
        raw_vision.start()
        high_level_vision.start()
        # Wait for processes in main process to terminate
        raw_vision.join()
        high_level_vision.join()
    except KeyboardInterrupt:
        print('KeyboardInterrupt :: Exiting gracefully...')
        high_level_vision.terminate()
        high_level_vision.join()
        print('High level vision process terminated and joined!')
        raw_vision.terminate()
        raw_vision.join()
        print('Raw vision process terminated and joined!')
        raw_vision_stream.close()
        high_level_vision_stream.close()
        print('Vision streams closed!')

