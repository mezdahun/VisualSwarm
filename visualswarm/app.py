"""
@author: mezdahun
@description: Main app of visualswarm
"""

import logging
from multiprocessing import Process, Queue
from visualswarm import env
from visualswarm.vision import vacquire, vprocess
from visualswarm.contrib import logparams

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)
bcolors = logparams.BColors


def health():
    """Entrypoint to start high level application"""
    logger.info("VisualSwarm application OK!")


def start_vision_stream():
    """Start the visual stream of the Pi"""
    logger.info(f'{bcolors.OKGREEN}START vision stream{bcolors.ENDC} ')
    raw_vision_stream = Queue()
    high_level_vision_stream = Queue()
    raw_vision = Process(target=vacquire.raw_vision, args=(raw_vision_stream,))
    high_level_vision = Process(target=vprocess.high_level_vision, args=(raw_vision_stream, high_level_vision_stream,))
    try:
        logger.info(f'{bcolors.OKBLUE}START{bcolors.ENDC} raw vision process')
        raw_vision.start()
        logger.info(f'{bcolors.OKBLUE}START{bcolors.ENDC} high level vision process')
        high_level_vision.start()
        # Wait for subprocesses in main process to terminate
        raw_vision.join()
        high_level_vision.join()
    except KeyboardInterrupt:
        logger.info(f'{bcolors.WARNING}EXIT gracefully on KeyboardInterrupt{bcolors.ENDC}')
        high_level_vision.terminate()
        high_level_vision.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} high level vision process and joined!')
        raw_vision.terminate()
        raw_vision.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} Raw vision process and joined!')
        raw_vision_stream.close()
        high_level_vision_stream.close()
        logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} vision streams!')
        logger.info(f'{bcolors.OKGREEN}EXITED Gracefully. Bye bye!{bcolors.ENDC}')
