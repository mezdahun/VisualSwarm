import logging
import threading
from queue import Queue
import sys

from visualswarm import env
from visualswarm.vision import vacquire, vprocess
from visualswarm.contrib import logparams, vision, simulation

from visualswarm.behavior import behavior

from visualswarm.control import motoroutput

from freezegun import freeze_time
import datetime
from time import sleep

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)
bcolors = logparams.BColors


def test_reader(sensor_stream, motor_set_stream):
    while True:
        prox_vals = sensor_stream.get()
        motor_set_stream.put(prox_vals[0])
        print(f'put done: {prox_vals[0]}')


def webots_interface(robot, sensors, motors, timestep, with_control=False):
    logger.info(f'Started VSWRM-Webots interface app with timestep: {timestep}')
    simulation_start_time = '2000-01-01 12:00:01'
    logger.info(f'Freezing time to: {simulation_start_time}')
    with freeze_time(simulation_start_time) as freezer:
        # sensor and motor value queues shared across subprocesses
        sensor_stream = Queue()
        motor_get_stream = Queue()
        motor_set_stream = Queue()

        # vision
        raw_vision_stream = Queue()
        high_level_vision_stream = Queue()
        VPF_stream = Queue()

        # interactive
        if vision.SHOW_VISION_STREAMS:
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

        # control
        control_stream = Queue()
        motor_control_mode_stream = Queue()
        emergency_stream = Queue()

        # A process to read and act according to sensor values
        raw_vision = threading.Thread(target=vacquire.simulated_vision, args=(raw_vision_stream,))
        high_level_vision_pool = [threading.Thread(target=vprocess.high_level_vision,
                                                   args=(raw_vision_stream,
                                                         high_level_vision_stream,
                                                         visualization_stream,
                                                         target_config_stream,)) for i in
                                  range(vision.NUM_SEGMENTATION_PROCS)]
        visualizer = threading.Thread(target=vprocess.visualizer, args=(visualization_stream, target_config_stream,))
        VPF_extractor = threading.Thread(target=vprocess.VPF_extraction, args=(high_level_vision_stream, VPF_stream,))
        behavior_proc = threading.Thread(target=behavior.VPF_to_behavior, args=(VPF_stream, control_stream,
                                                                                motor_control_mode_stream, with_control))
        motor_control = threading.Thread(target=motoroutput.control_thymio, args=(control_stream, motor_control_mode_stream,
                                                                                  emergency_stream, with_control,
                                                                                  motor_set_stream))
        emergency_proc = threading.Thread(target=motoroutput.emergency_behavior, args=(emergency_stream, sensor_stream))

        # Start subprocesses
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision process')
        raw_vision.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} high level vision processes')
        for proc in high_level_vision_pool:
            logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} start---')
            proc.start()
        visualizer.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} VPF extractor process')
        VPF_extractor.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} behavior process')
        behavior_proc.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} control process')
        motor_control.start()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} sentinel/emergency process')
        emergency_proc.start()

        # The main thread to interact with non-pickleable objects that can not be passed
        # to subprocesses
        sensor_get_time = 0  # virtual time increment in ms
        while robot.step(timestep) != -1:
            # logger.info(f'sensor_get_time: {sensor_get_time}')
            # Thymio updates sensor values on predefined frequency irr
            if (sensor_get_time / 1000) > (1 / simulation.UPFREQ_PROX_HORIZONTAL):
                prox_vals = [i.getValue() for i in sensors['prox']['horizontal']]
                try:
                    sensor_stream.get_nowait()
                except:
                    pass
                sensor_stream.put(prox_vals)
                sensor_get_time = sensor_get_time % (1 / simulation.UPFREQ_PROX_HORIZONTAL)

            if motor_set_stream.qsize() > 0:
                motor_vals = motor_set_stream.get()
                logger.info(motor_vals)
                motors['left'].setVelocity(motor_vals['left'] / 100)
                motors['right'].setVelocity(motor_vals['right'] / 100)

            # increment virtual time counters
            sensor_get_time += timestep

            # ticking virtual time with virtual time of Webots environment
            freezer.tick(delta=datetime.timedelta(milliseconds=timestep))

            # sleeping with physical time so that all processes can calculate until the next simulation timestep
            sleep(0.01)

        # End subprocesses
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision process')
        raw_vision.join()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} high level vision processes')
        for proc in high_level_vision_pool:
            proc.join()
        visualizer.join()
        VPF_extractor.join()
        behavior_proc.join()

        sensor_stream.close()
        motor_get_stream.close()
        motor_set_stream.close()

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
