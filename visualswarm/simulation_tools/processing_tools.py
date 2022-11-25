"""
@author: mezdahun
@description: tools during WeBots simulation and processing
"""
import os
import numpy as np
import logging

from visualswarm import env
from visualswarm.vision import vprocess
from visualswarm.contrib import logparams, vision, simulation, monitoring
from visualswarm.behavior import behavior
from visualswarm.control import motoroutput

if simulation.SPARE_RESCOURCES:
    from threading import Thread
    from queue import Queue
else:
    from multiprocessing import Process, Queue

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(monitoring.LOG_LEVEL)
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


def return_processing_streams():
    # sensor and motor value queues shared across subprocesses

    streams = []

    for i in range(6):
        streams.append(Queue())

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

    streams.append(visualization_stream)
    streams.append(target_config_stream)

    # control
    for i in range(3):
        streams.append(Queue())

    return streams


def return_processes(streams, with_control=False):
    [sensor_stream, motor_get_stream, webots_do_stream,
     raw_vision_stream, high_level_vision_stream, VPF_stream,
     visualization_stream, target_config_stream,
     control_stream, motor_control_mode_stream, emergency_stream] = streams

    # A process to read and act according to sensor values
    high_level_vision_pool = [VSWRMParallelObject(target=vprocess.high_level_vision,
                                                  args=(raw_vision_stream,
                                                        high_level_vision_stream,
                                                        visualization_stream,
                                                        target_config_stream,)) for i in
                              range(1)]
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
    return [high_level_vision_pool, visualizer, VPF_extractor, behavior_proc, motor_control, emergency_proc]


def start_processes(processes):
    [high_level_vision_pool, visualizer, VPF_extractor,
     behavior_proc, motor_control, emergency_proc] = processes

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


def assure_data_folders(robot_name):
    save_folder = os.path.join(simulation.WEBOTS_SIM_SAVE_FOLDER, robot_name)

    # creating main data folder
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        run_num = 1
    else:
        subfolders = [name for name in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, name))]
        subfolders = [int(name) for name in subfolders]

        if len(subfolders) == 0:
            run_num = 1
        else:
            run_num = np.max(subfolders) + 1

    save_folder = os.path.join(save_folder, str(run_num))
    os.makedirs(save_folder)

    filename_or = os.path.join(save_folder, f'{robot_name}_run{run_num}_or.npy')
    filename_pos = os.path.join(save_folder, f'{robot_name}_run{run_num}_pos.npy')
    filename_params = os.path.join(save_folder, f'{robot_name}_run{run_num}_params.json')

    return filename_pos, filename_or, filename_params, run_num


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def robot_orientation(compass_values):
    """ The zero direction can be given as [X, Y, Z] in contrib.simulation"""
    zero_vector = np.array(simulation.WEBOTS_ZERO_ORIENTATION)
    zero_vector = zero_vector[~np.isnan(zero_vector)]

    v1_u = unit_vector(zero_vector)

    compass_values = np.array(compass_values)
    compass_values = compass_values[~np.isnan(compass_values)]

    v2_u = unit_vector(compass_values)

    angle = np.sign(compass_values[simulation.WEBOTS_ORIENTATION_SIGN_IDX]) * np.arccos(np.dot(v1_u, v2_u))
    return angle


def stop_and_cleanup(processes, streams):
    [sensor_stream, motor_get_stream, webots_do_stream,
     raw_vision_stream, high_level_vision_stream, VPF_stream,
     visualization_stream, target_config_stream,
     control_stream, motor_control_mode_stream, emergency_stream] = streams
    [high_level_vision_pool, visualizer, VPF_extractor,
     behavior_proc, motor_control, emergency_proc] = processes
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