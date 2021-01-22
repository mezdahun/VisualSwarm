"""
@author: mezdahun
@description: Main app of visualswarm
"""

import logging
from multiprocessing import Process, Queue
from visualswarm import env
from visualswarm.monitoring import ifdb, system_monitor
from visualswarm.vision import vacquire, vprocess
from visualswarm.contrib import logparams, segmentation, visual
from visualswarm.behavior import control, motoroutput
import time

import dbus.mainloop.glib
from gi.repository import GObject as gobject
from gi.repository import GLib

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

    gobject.threads_init()

    loop = GLib.MainLoop()

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SessionBus()

    # Create Aseba network
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                             dbus_interface='ch.epfl.mobots.AsebaNetwork')

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
    visualizer.start()
    VPF_extractor = Process(target=vprocess.VPF_extraction, args=(high_level_vision_stream, VPF_stream,))
    behavior = Process(target=control.VPF_to_behavior, args=(VPF_stream, control_stream,))
    motor_control = Process(target=motoroutput.control_thymio, args=(control_stream, network))
    system_monitor_proc = Process(target=system_monitor.system_monitor)

    try:
        # Start subprocesses
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision process')
        raw_vision.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} high level vision processes')
        for proc in high_level_vision_pool:
            proc.start()
        # visualizer.start()
        VPF_extractor.start()
        behavior.start()
        motor_control.start()
        system_monitor_proc.start()

        # Wait for subprocesses in main process to terminate
        visualizer.join()
        for proc in high_level_vision_pool:
            proc.join()
        raw_vision.join()
        VPF_extractor.join()
        behavior.join()
        motor_control.join()
        system_monitor_proc.join()

    except KeyboardInterrupt:
        logger.info(f'{bcolors.WARNING}EXIT gracefully on KeyboardInterrupt{bcolors.ENDC}')

        # Terminating Processes
        system_monitor_proc.terminate()
        system_monitor_proc.join()
        behavior.terminate()
        behavior.join()
        control_stream.put((0, 0))
        time.sleep(0.2)
        motor_control.terminate()
        motor_control.join()
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

        network.SetVariable("thymio-II", "motor.left.target", [0])
        network.SetVariable("thymio-II", "motor.right.target", [0])
