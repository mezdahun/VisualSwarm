"""
@author: mezdahun
@description: Main app of visualswarm
"""

import logging
from multiprocessing import Process, Queue
from visualswarm import env
from visualswarm.vision import vacquire, vprocess
from visualswarm.contrib import logparams, segmentation
import cv2

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
    if segmentation.FIND_COLOR_INTERACTIVE:
        target_config_stream = Queue()
    else:
        target_config_stream = None
    raw_vision = Process(target=vacquire.raw_vision, args=(raw_vision_stream,))

    high_level_vision_pool = [Process(target=vprocess.high_level_vision, args=(raw_vision_stream, high_level_vision_stream, target_config_stream,)) for i in range(segmentation.NUM_SEGMENTATION_PROCS)]
    visualizer = Process(target=vprocess.visualizer, args=(high_level_vision_stream, target_config_stream,))
    try:
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision process')
        raw_vision.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} high level vision processes')
        for proc in high_level_vision_pool:
            proc.start()
        visualizer.start()
        # Wait for subprocesses in main process to terminate
        visualizer.join()
        for proc in high_level_vision_pool:
            proc.join()
        raw_vision.join()
    except KeyboardInterrupt:
        logger.info(f'{bcolors.WARNING}EXIT gracefully on KeyboardInterrupt{bcolors.ENDC}')
        visualizer.terminate()
        visualizer.join()
        for proc in high_level_vision_pool:
            proc.terminate()
            proc.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} high level vision process and joined!')
        raw_vision.terminate()
        raw_vision.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} Raw vision process and joined!')
        raw_vision_stream.close()
        high_level_vision_stream.close()
        logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} vision streams!')
        logger.info(f'{bcolors.OKGREEN}EXITED Gracefully. Bye bye!{bcolors.ENDC}')
