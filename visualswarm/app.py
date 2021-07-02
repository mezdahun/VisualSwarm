"""
@author: mezdahun
@description: Main app of visualswarm
"""

from multiprocessing import Process, Queue
import sys
import signal
import time

import visualswarm.contrib.vision
from visualswarm import env
from visualswarm.monitoring import ifdb, drive_uploader  # system_monitor
from visualswarm.vision import vacquire, vprocess
from visualswarm.contrib import logparams, vision, simulation, monitoring
from visualswarm.behavior import behavior
from visualswarm.control import motorinterface, motoroutput

if not simulation.ENABLE_SIMULATION:
    import dbus.mainloop.glib
    dbus.mainloop.glib.threads_init()

signal.signal(signal.SIGINT, signal.default_int_handler)

# setup logging
import os
ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')

if monitoring.ENABLE_CLOUD_LOGGING:
    import google.cloud.logging
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = monitoring.GOOGLE_APPLICATION_CREDENTIALS
    # Instantiates a client
    client = google.cloud.logging.Client()
    client.get_default_handler()
    client.setup_logging()

import logging
logging.basicConfig()
logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
logger.setLevel(monitoring.LOG_LEVEL)
bcolors = logparams.BColors


def health():
    """Entrypoint to start high level application"""
    logger.info("VisualSwarm application OK!")


def start_application(with_control=False):
    """Start the visual stream of the Pi"""
    # Starting fresh database if requested
    if env.INFLUX_FRESH_DB_UPON_START:
        logger.info(f'{bcolors.OKGREEN}CLEAN InfluxDB{bcolors.ENDC} upon start as requested')
        ifclient = ifdb.create_ifclient()
        ifclient.drop_database(env.INFLUX_DB_NAME)
        ifclient.create_database(env.INFLUX_DB_NAME)

    logger.info(f'{bcolors.OKGREEN}START vision stream{bcolors.ENDC} ')

    # connect to Thymio
    if with_control:
        motorinterface.asebamedulla_init()

    # Creating Queues
    raw_vision_stream = Queue()
    high_level_vision_stream = Queue()

    if vision.SHOW_VISION_STREAMS or monitoring.SAVE_VISION_VIDEO:
        # showing raw and processed camera stream
        visualization_stream = Queue()
    else:
        visualization_stream = None

    if vision.FIND_COLOR_INTERACTIVE:
        # interactive target parameter tuning turned on
        target_config_stream = Queue()
        # overriding visualization otherwise interactive parameter tuning makes no sense
        visualization_stream = Queue()
    else:
        target_config_stream = None

    VPF_stream = Queue()
    control_stream = Queue()
    motor_control_mode_stream = Queue()
    emergency_stream = Queue()

    # Creating main processes
    raw_vision = Process(target=vacquire.raw_vision, args=(raw_vision_stream,))

    if vision.RECOGNITION_TYPE=="Color":
        vp_target = vprocess.high_level_vision
    elif vision.RECOGNITION_TYPE=="CNN":
        vp_target = vprocess.high_level_vision_CNN
    high_level_vision_pool = [Process(target=vp_target,
                                      args=(raw_vision_stream,
                                            high_level_vision_stream,
                                            visualization_stream,
                                            target_config_stream,)) for i in range(
        visualswarm.contrib.vision.NUM_SEGMENTATION_PROCS)]

    visualizer = Process(target=vprocess.visualizer, args=(visualization_stream, target_config_stream,))
    VPF_extractor = Process(target=vprocess.VPF_extraction, args=(high_level_vision_stream, VPF_stream,))
    behavior_proc = Process(target=behavior.VPF_to_behavior, args=(VPF_stream, control_stream,
                                                                   motor_control_mode_stream, with_control))
    motor_control = Process(target=motoroutput.control_thymio, args=(control_stream, motor_control_mode_stream,
                                                                     emergency_stream, with_control))
    # system_monitor_proc = Process(target=system_monitor.system_monitor)
    emergency_proc = Process(target=motoroutput.emergency_behavior, args=(emergency_stream,))

    try:
        # Start subprocesses
        if vision.RECOGNITION_TYPE == "Color":
            logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision in separate process')
            raw_vision.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} high level vision processes')
        for proc in high_level_vision_pool:
            proc.start()
            time.sleep(0.5)
        visualizer.start()
        VPF_extractor.start()
        behavior_proc.start()
        motor_control.start()
        # system_monitor_proc.start()
        if with_control:
            emergency_proc.start()

        # Wait for subprocesses in main process to terminate
        visualizer.join()
        for proc in high_level_vision_pool:
            proc.join()
        if vision.RECOGNITION_TYPE == "Color":
            raw_vision.join()
        VPF_extractor.join()
        behavior_proc.join()
        motor_control.join()
        # system_monitor_proc.join()
        if with_control:
            emergency_proc.join()

    except KeyboardInterrupt:
        # suppressing all error messages during graceful exit that come from intermingled queues
        sys.stderr = object

        logger.info(f'{bcolors.WARNING}EXIT gracefully on KeyboardInterrupt{bcolors.ENDC}')

        # Terminating Processes
        if with_control:
            emergency_proc.terminate()
            emergency_proc.join()
            logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} emergency process and joined!')
        # system_monitor_proc.terminate()
        # system_monitor_proc.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} system monitor process and joined!')
        motor_control.terminate()
        motor_control.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} motor control process and joined!')
        behavior_proc.terminate()
        behavior_proc.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} control parameter calculations!')
        VPF_extractor.terminate()
        VPF_extractor.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} visual field segmentation!')
        visualization_stream.put('RELEASE AND TERMINATE')
        time.sleep(5)
        visualizer.terminate()
        visualizer.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} visualization stream!')
        for proc in high_level_vision_pool:
            proc.terminate()
            proc.join()
        logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} high level vision process(es) and joined!')
        if vision.RECOGNITION_TYPE == "Color":
            raw_vision.terminate()
            raw_vision.join()
            logger.info(f'{bcolors.WARNING}TERMINATED{bcolors.ENDC} Raw vision process and joined!')

        # Closing Queues
        raw_vision_stream.close()
        high_level_vision_stream.close()
        VPF_stream.close()
        logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} vision streams!')
        if visualization_stream is not None:
            visualization_stream.close()
            logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} visualization stream!')
        if target_config_stream is not None:
            target_config_stream.close()
            logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} configuration stream!')
        control_stream.close()
        logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} control parameter stream!')
        motor_control_mode_stream.close()
        logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} movement mode stream!')
        emergency_stream.close()
        logger.info(f'{bcolors.WARNING}CLOSED{bcolors.ENDC} emergency stream!')

        if with_control:
            logger.info(f'{bcolors.OKGREEN}Setting Thymio2 velocity to zero...{bcolors.ENDC}')
            dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
            bus = dbus.SessionBus()
            network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                     dbus_interface='ch.epfl.mobots.AsebaNetwork')
            network.SetVariable("thymio-II", "motor.left.target", [0])
            network.SetVariable("thymio-II", "motor.right.target", [0])
            motoroutput.light_up_led(network, 0, 0, 0)
            motorinterface.asebamedulla_end()

        if monitoring.ENABLE_CLOUD_STORAGE:
            logger.info(f'{bcolors.OKGREEN}UPLOAD{bcolors.ENDC} generated videos to Google Drive...')
            drive_uploader.upload_vision_videos(monitoring.SAVED_VIDEO_FOLDER)

        logger.info(f'{bcolors.OKGREEN}EXITED Gracefully. Bye bye!{bcolors.ENDC}')


def start_application_with_control():
    start_application(with_control=True)

