import logging
import numpy as np
import pickle
from contextlib import ExitStack

from visualswarm import env
from visualswarm.contrib import logparams, simulation, control
from visualswarm.simulation_tools import processing_tools

from freezegun import freeze_time
import datetime
import time

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)
bcolors = logparams.BColors


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

        camera_data = devices['camera'].getImage()
        img = np.frombuffer(camera_data, np.uint8).reshape((height, width, 4))
        return img


def webots_entrypoint(robot, devices, timestep, with_control=False):
    logger.info(f'Started VSWRM-Webots interface app with timestep: {timestep}')
    simulation_start_time = '2000-01-01 12:00:01'
    logger.info(f'Freezing time to: {simulation_start_time}')
    with freeze_time(simulation_start_time) as freezer:

        # sensor and motor value queues shared across subprocesses
        streams = processing_tools.return_processing_streams()
        # creating subprocesses
        processes = processing_tools.return_processes(streams, with_control=with_control)
        # get streams that need to be filled from main thread
        [sensor_stream, _, webots_do_stream, raw_vision_stream, _, _, _, _, _, _, _] = streams

        # start all subprocesses
        processing_tools.start_processes(processes)

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
            position_fpath, orientation_fpath = processing_tools.assure_data_folders(robot.getName())

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
                                                        np.array([processing_tools.robot_orientation(raw_or_vals)])))
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

        processing_tools.stop_and_cleanup(processes, streams)
