"""
@author: mezdahun
@description: Main app of visualswarm
"""

import logging
from multiprocessing import Process, Queue
from visualswarm import env
from visualswarm.monitoring import ifdb
from visualswarm.vision import vacquire, vprocess
from visualswarm.contrib import logparams, segmentation, visual
from visualswarm.behavior import control

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
    # Starting fresh database if requested
    if env.INFLUX_FRESH_DB_UPON_START:
        logger.info(f'{bcolors.OKGREEN}CLEAN InfluxDB{bcolors.ENDC} upon start as requested')
        ifclient = ifdb.create_ifclient()
        ifclient.drop_database(env.INFLUX_DB_NAME)
        ifclient.create_database(env.INFLUX_DB_NAME)

    logger.info(f'{bcolors.OKGREEN}START vision stream{bcolors.ENDC} ')

    # Creating Queues
    raw_vision_stream = Queue()
    high_level_vision_stream = Queue()

    if visual.SHOW_VISION_STREAMS:
        # showing raw and processed camera stream
        visualization_stream = Queue()
    else:
        visualization_stream = None

    if visual.FIND_COLOR_INTERACTIVE:
        # interactive target parameter tuning turned on
        target_config_stream = Queue()
        # overriding visualization otherwise interactive parameter tuning makes no sense
        visualization_stream = Queue()
    else:
        target_config_stream = None

    VPF_stream = Queue()
    control_stream = Queue()

    # Creating main processes
    raw_vision = Process(target=vacquire.raw_vision, args=(raw_vision_stream,))
    high_level_vision_pool = [Process(target=vprocess.high_level_vision,
                                      args=(raw_vision_stream,
                                            high_level_vision_stream,
                                            visualization_stream,
                                            target_config_stream,)) for i in range(segmentation.NUM_SEGMENTATION_PROCS)]
    visualizer = Process(target=vprocess.visualizer, args=(visualization_stream, target_config_stream,))
    VPF_extractor = Process(target=vprocess.VPF_extraction, args=(high_level_vision_stream, VPF_stream,))
    behavior = Process(target=control.VPF_to_behavior, args=(VPF_stream, control_stream,))

    try:
        # Start subprocesses
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision process')
        raw_vision.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} high level vision processes')
        for proc in high_level_vision_pool:
            proc.start()
        visualizer.start()
        VPF_extractor.start()
        behavior.start()

        # Wait for subprocesses in main process to terminate
        visualizer.join()
        for proc in high_level_vision_pool:
            proc.join()
        raw_vision.join()
        VPF_extractor.join()
        behavior.join()

    except KeyboardInterrupt:
        logger.info(f'{bcolors.WARNING}EXIT gracefully on KeyboardInterrupt{bcolors.ENDC}')

        # Terminating Processes
        behavior.terminate()
        behavior.join()
        VPF_extractor.terminate()
        VPF_extractor.join()
        visualizer.terminate()
        visualizer.join()
        for proc in high_level_vision_pool:
            proc.terminate()
            proc.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} high level vision process and joined!')
        raw_vision.terminate()
        raw_vision.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} Raw vision process and joined!')

        # Closing Queues
        raw_vision_stream.close()
        high_level_vision_stream.close()
        if visualization_stream is not None:
            visualization_stream.close()
        if target_config_stream is not None:
            target_config_stream.close()
        VPF_stream.close()
        control_stream.close()
        logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} vision streams!')
        logger.info(f'{bcolors.OKGREEN}EXITED Gracefully. Bye bye!{bcolors.ENDC}')
