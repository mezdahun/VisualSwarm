import logging
import numpy as np
from cv2 import rotate, flip, cvtColor, ROTATE_90_CLOCKWISE, COLOR_BGR2RGB

from visualswarm import env
from visualswarm.vision import vacquire, vprocess
from visualswarm.contrib import logparams, vision, simulation, control
from visualswarm.behavior import behavior
from visualswarm.control import motoroutput

if simulation.SPARE_RESCOURCES:
    from threading import Thread
    from queue import Queue
else:
    from multiprocessing import Process, Queue

from freezegun import freeze_time
import datetime
from time import sleep

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)
bcolors = logparams.BColors


class VSWRMParallelObject(object):
    """Wrapper around runnable objects so we can switch between Threads and Processes according to the
    architecture we run the simulation on"""

    def __init__(self, target=None, args=None):
        if simulation.SPARE_RESCOURCES:
            self.runnable = Thread(target=target, args=args)
            self.runnable_type = Thread
        else:
            self.runnable = Process(target=target, args=args)
            self.runnable_type = Process

    def start(self):
        self.runnable.start()

    def terminate(self):
        if self.runnable_type == Process:
            # Only Process has terminate, threads will just join
            self.runnable.terminate()

    def join(self):
        self.runnable.join()


def webots_do(control_args, devices):
    command = control_args[0]
    command_arg = control_args[1]
    MAX_WEBOTS_MOTOR_SPEED = 9.53
    if command == "SET_MOTOR":
        logger.debug(f"webots_do_move: {command_arg['left']} * {MAX_WEBOTS_MOTOR_SPEED} / {control.MAX_MOTOR_SPEED}")
        devices['motors']['left'].setVelocity(command_arg['left'] * (MAX_WEBOTS_MOTOR_SPEED/control.MAX_MOTOR_SPEED))
        devices['motors']['right'].setVelocity(command_arg['right'] * (MAX_WEBOTS_MOTOR_SPEED/control.MAX_MOTOR_SPEED))
    elif command == "LIGHTUP_LED":
        devices['leds']['top'].set(command_arg)


def webots_entrypoint(robot, sensors, devices, timestep, with_control=False):
    logger.info(f'Started VSWRM-Webots interface app with timestep: {timestep}')
    simulation_start_time = '2000-01-01 12:00:01'
    logger.info(f'Freezing time to: {simulation_start_time}')
    with freeze_time(simulation_start_time) as freezer:
        # sensor and motor value queues shared across subprocesses
        sensor_stream = Queue()
        motor_get_stream = Queue()
        webots_do_stream = Queue()

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
        high_level_vision_pool = [VSWRMParallelObject(target=vprocess.high_level_vision,
                                                      args=(raw_vision_stream,
                                                            high_level_vision_stream,
                                                            visualization_stream,
                                                            target_config_stream,)) for i in
                                  range(vision.NUM_SEGMENTATION_PROCS)]
        visualizer = VSWRMParallelObject(target=vprocess.visualizer, args=(visualization_stream, target_config_stream,))
        VPF_extractor = VSWRMParallelObject(target=vprocess.VPF_extraction,
                                            args=(high_level_vision_stream, VPF_stream,))
        behavior_proc = VSWRMParallelObject(target=behavior.VPF_to_behavior, args=(VPF_stream, control_stream,
                                                                                   motor_control_mode_stream,
                                                                                   with_control))
        motor_control = VSWRMParallelObject(target=motoroutput.control_thymio,
                                            args=(control_stream, motor_control_mode_stream,
                                                  emergency_stream, with_control,
                                                  webots_do_stream))
        emergency_proc = VSWRMParallelObject(target=motoroutput.emergency_behavior,
                                             args=(emergency_stream, sensor_stream))

        # Start subprocesses
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
        camera_sampling_period = devices['camera'].getSamplingPeriod()
        camera_get_time = 0
        frame_id = 0

        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision from main thread/process')
        while robot.step(timestep) != -1:

            # Fetching camera image on a predefined frequency
            if camera_get_time > camera_sampling_period:
                logger.info(f'capturing frame_id: {frame_id}')
                try:
                    raw_vision_stream.get_nowait()
                except:
                    pass
                img = flip(rotate(np.array(devices['camera'].getImageArray(),
                                           dtype=np.uint8),
                                  ROTATE_90_CLOCKWISE), 1)
                img = cvtColor(img, COLOR_BGR2RGB)
                raw_vision_stream.put((img, frame_id, datetime.datetime.utcnow()))

                frame_id += 1
                camera_get_time = camera_get_time % (1 / simulation.UPFREQ_PROX_HORIZONTAL)

            # Thymio updates sensor values on predefined frequency
            if (sensor_get_time / 1000) > (1 / simulation.UPFREQ_PROX_HORIZONTAL):
                prox_vals = [i.getValue() for i in sensors['prox']['horizontal']]
                try:
                    sensor_stream.get_nowait()
                except:
                    pass
                sensor_stream.put(prox_vals)
                sensor_get_time = sensor_get_time % (1 / simulation.UPFREQ_PROX_HORIZONTAL)

            if webots_do_stream.qsize() > 0:
                # motor_vals = motoroutput.get_latest_element(webots_do_stream)
                command_set = webots_do_stream.get()
                logger.debug(command_set)
                webots_do(command_set, devices)
                # devices['motors']['left'].setVelocity(motor_vals[1]['left'] / 100)
                # devices['motors']['right'].setVelocity(motor_vals[1]['right'] / 100)

            # increment virtual time counters
            sensor_get_time += timestep
            camera_get_time += timestep

            # ticking virtual time with virtual time of Webots environment
            freezer.tick(delta=datetime.timedelta(milliseconds=timestep))

            # sleeping with physical time so that all processes can calculate until the next simulation timestep
            # sleep(0.01)

        # End subprocesses
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision process')
        # raw_vision.join()
        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} high level vision processes')
        for proc in high_level_vision_pool:
            proc.terminate()
            proc.join()
        visualizer.terminate()
        visualizer.join()
        VPF_extractor.terminate()
        VPF_extractor.join()
        behavior_proc.terminate()
        behavior_proc.join()
        motor_control.terminate()
        motor_control.join()
        emergency_proc.terminate()
        emergency_proc.join()

        sensor_stream.close()
        motor_get_stream.close()
        webots_do_stream.close()

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
