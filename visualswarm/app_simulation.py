import logging
import numpy as np
import os
import pickle
from contextlib import ExitStack

from visualswarm import env
from visualswarm.vision import vprocess
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
import time

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

    return filename_pos, filename_or

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


def webots_do(control_args, devices):
    command = control_args[0]
    command_arg = control_args[1]
    if command == "SET_MOTOR":
        v_left = command_arg['left'] * (simulation.MAX_WEBOTS_MOTOR_SPEED / control.MAX_MOTOR_SPEED)
        v_right = command_arg['right'] * (simulation.MAX_WEBOTS_MOTOR_SPEED / control.MAX_MOTOR_SPEED)
        logger.debug(f"webots_do move: left {v_left}, right {v_right}")
        devices['motors']['left'].setVelocity(v_left)
        devices['motors']['right'].setVelocity(v_right)
    elif command == "LIGHTUP_LED":
        logger.debug(f"webots_do light: color {command_arg}")
        devices['leds']['top'].set(command_arg)


def getWebotsCameraImage(devices):
        rawString = devices['camera'].getImage()
        width = devices['params']['c_width']
        height = devices['params']['c_height']

        # # Create mask for yellow pixels based on the camera image.
        # index = 0
        # img = np.zeros([height, width, 3], np.uint8)
        # for j in range(height):
        #     for i in range(width):
        #         img[j, i, 2] = rawString[index]
        #         img[j, i, 1] = rawString[index + 1]
        #         img[j, i, 0] = rawString[index + 2]
        #
        # return img

        camera_data = devices['camera'].getImage()

        # solution1
        # camera_image = decode_image(camera_data, height, width)
        # print(camera_image.shape)

        # #solution2
        # nparr = np.frombuffer(camera_data, np.uint8)
        # slicer_idx = int(len(nparr) / 4)
        # # r = nparr[0:slicer_idx].reshape(height, width)
        # # g = nparr[slicer_idx:2 * slicer_idx].reshape(height, width)
        # # b = nparr[2 * slicer_idx:3 * slicer_idx].reshape(height, width)
        # r = nparr[0:3 * slicer_idx:3].reshape(height, width)
        # g = nparr[1:3 * slicer_idx + 1:3].reshape(height, width)
        # b = nparr[2:3 * slicer_idx + 2:3].reshape(height, width)
        # img = np.dstack((r, g, b))

        img = np.frombuffer(camera_data, np.uint8).reshape((height, width, 4))

        return img


def webots_entrypoint(robot, devices, timestep, with_control=False):
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

        # to control frquency of given actions we use counters with simulation time
        sensor_get_time = 0
        camera_sampling_period = devices['camera'].getSamplingPeriod()
        camera_get_time = 0
        frame_id = 0
        simulation_time = 0  # keep up with virtual time in simulation (in ms)

        logger.info(f'{bcolors.OKGREEN}START{bcolors.ENDC} raw vision (webots) from main thread/process')
        avg_times = np.zeros(5)
        timer_counts = np.zeros(5)
        t_end = time.perf_counter()

        # getting filenames to save simulation data and creating folders
        if simulation.WEBOTS_SAVE_SIMULATION_DATA:
            position_fpath, orientation_fpath = assure_data_folders(robot.getName())

        with ExitStack() if not simulation.WEBOTS_SAVE_SIMULATION_DATA else open(position_fpath, 'ab') as pos_f:
            with ExitStack() if not simulation.WEBOTS_SAVE_SIMULATION_DATA else open(orientation_fpath, 'ab') as or_f:
                while robot.step(timestep) != -1:

                    t_start = time.perf_counter()
                    dt_tick = t_start - t_end
                    avg_times[4] += dt_tick
                    timer_counts[4] += 1

                    t0 = time.perf_counter()

                    # saving simulation data if requested
                    if simulation.WEBOTS_SAVE_SIMULATION_DATA:

                        raw_or_vals = devices['monitor']['orientation'].getValues()
                        r_orientation = np.concatenate((np.array([simulation_time]),
                                                        np.array([robot_orientation(raw_or_vals)])))
                        r_position = np.concatenate((np.array([simulation_time]),
                                                     np.array(devices['monitor']['gps'].getValues())))

                        pickle.dump(r_position, pos_f)
                        pickle.dump(r_orientation, or_f)

                    t1 = time.perf_counter()
                    dt_save = t1-t0
                    avg_times[0] += dt_save
                    timer_counts[0] += 1

                    # Fetching camera image on a predefined frequency
                    if camera_get_time > camera_sampling_period:
                        t0 = time.perf_counter()

                        try:
                            raw_vision_stream.get_nowait()
                        except:
                            pass

                        img = getWebotsCameraImage(devices)
                        raw_vision_stream.put((img, frame_id, datetime.datetime.utcnow()))

                        frame_id += 1
                        camera_get_time = camera_get_time % (1 / simulation.UPFREQ_PROX_HORIZONTAL)

                        t1 = time.perf_counter()
                        dt_image = t1 - t0
                        avg_times[1] += dt_image
                        timer_counts[1] += 1

                    # Thymio updates sensor values on predefined frequency
                    if (sensor_get_time / 1000) > (1 / simulation.UPFREQ_PROX_HORIZONTAL):
                        t0 = time.perf_counter()

                        prox_vals = [i.getValue() for i in devices['sensors']['prox']['horizontal']]
                        try:
                            sensor_stream.get_nowait()
                        except:
                            pass
                        sensor_stream.put(prox_vals)
                        sensor_get_time = sensor_get_time % (1 / simulation.UPFREQ_PROX_HORIZONTAL)

                        t1 = time.perf_counter()
                        dt_prox = t1 - t0
                        avg_times[2] += dt_prox
                        timer_counts[2] += 1

                    # Acting on robot devices according to controller
                    if webots_do_stream.qsize() > 0:
                        t0 = time.perf_counter()

                        command_set = webots_do_stream.get_nowait()
                        logger.debug(command_set)
                        webots_do(command_set, devices)

                        t1 = time.perf_counter()
                        dt_webotsdo = t1 - t0
                        avg_times[3] += dt_webotsdo
                        timer_counts[3] += 1

                    # increment virtual time counters
                    sensor_get_time += timestep
                    camera_get_time += timestep
                    simulation_time += timestep

                    # ticking virtual time with virtual time of Webots environment
                    freezer.tick(delta=datetime.timedelta(milliseconds=timestep))

                    t_end = time.perf_counter()
                    used_times = avg_times/timer_counts*1000
                    logger.info(f'\nAVG Used times: \n'
                                f'\t---data save: {used_times[0]} \n'
                                f'\t---image passing: {used_times[1]}\n'
                                f'\t---sensor passing: {used_times[2]}\n'
                                f'\t---set devices: {used_times[3]}\n'
                                f'\t---step world physics: {used_times[4]}\n')

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
