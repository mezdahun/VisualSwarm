"""
@author: mezdahun
@description: Submodule to implement control according to visual stream
"""
import datetime
import logging

import numpy as np

import visualswarm.contrib.vision
from visualswarm.monitoring import ifdb
from visualswarm.contrib import monitoring, simulation, control
from visualswarm.contrib import algorithm_improvements as algoimp
from visualswarm.behavior import statevarcomp
from visualswarm import env
from queue import Empty

if monitoring.ENABLE_CLOUD_STORAGE:
    import pickle  # nosec

# using main logger
if not simulation.ENABLE_SIMULATION:
    # setup logging
    import os
    ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
    logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
    logger.setLevel(monitoring.LOG_LEVEL)
else:
    logger = logging.getLogger('visualswarm.app_simulation')  # pragma: simulation no cover

def get_latest_element(queue):  # pragma: simulation no cover
    """
    fetching the latest element in queue and by that emptying the FIFO Queue object. Use this to consume queue elements
    with a slow process that is filled up by a faster process
        Args:
            queue (multiprocessing.Queue): queue object to be emptied and returned the latest element
        Returns:
            val: latest vaue in the queue
    """
    val = None
    while not queue.empty():
        try:
            val = queue.get_nowait()
        except Empty:
            return val
    return val

def VPF_to_behavior(VPF_stream, control_stream, motor_control_mode_stream, with_control=False):
    """
    Process to extract final visual projection field from high level visual input.
        Args:
            VPF_stream (multiprocessing Queue): stream to receive visual projection field
            control_stream (multiprocessing Queue): stream to push calculated control parameters
            motor_control_mode_stream (multiprocessing Queue): stream to determine which movement regime the agent
                should follow
            with_control (boolean): if true, the output of the algorithm is sent to the movement processes.
        Returns:
            -shall not return-
    """
    try:
        if not simulation.ENABLE_SIMULATION:
            measurement_name = "control_parameters"
            ifclient = ifdb.create_ifclient()

        phi = None
        v = 0
        t_prev = datetime.datetime.now()
        is_initialized = False
        # start_behave = t_prev
        # prev_sign = 0

        element = None
        while element is None:
            element = get_latest_element(VPF_stream)

        (projection_field, capture_timestamp, projection_field_c2) = element

        if not visualswarm.contrib.vision.divided_projection_field:
            phi = np.linspace(visualswarm.contrib.vision.PHI_START, visualswarm.contrib.vision.PHI_END,
                              len(projection_field))
        else:
            phi = np.linspace(visualswarm.contrib.vision.PHI_START, visualswarm.contrib.vision.PHI_END,
                              projection_field.shape[0])

        ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
        EXP_ID = os.getenv('EXP_ID', 'expXXXXXX')
        statevar_timestamp = datetime.datetime.now().strftime("%d-%m-%y-%H%M%S")
        statevars_fpath = os.path.join(monitoring.SAVED_VIDEO_FOLDER, f'{statevar_timestamp}_{EXP_ID}_{ROBOT_NAME}_statevars.npy')
        if monitoring.ENABLE_CLOUD_STORAGE:
            os.makedirs(monitoring.SAVED_VIDEO_FOLDER, exist_ok=True)

        rw_dt = 0
        add_psi = 0.1
        new_dpsi = 0.05

        dpsi_before = None

        while True:
            try:
                # the possibility of 2 detection classes is already included, only using the first for now
                (projection_field, capture_timestamp, projection_field_c2) = get_latest_element(VPF_stream)
            except:
                continue

            if np.mean(projection_field) == 0:
                if control.EXP_MOVE_TYPE != 'NoExploration' or algoimp.WITH_EXPLORE_ROT or algoimp.WITH_EXPLORE_ROT_CONT:
                    movement_mode = "EXPLORE"
                else:
                    movement_mode = "BEHAVE"
            else:
                movement_mode = "BEHAVE"

            t_now = datetime.datetime.now()
            dt = (t_now - t_prev).total_seconds()  # to normalize

            if not visualswarm.contrib.vision.divided_projection_field:
                dv, dpsi = statevarcomp.compute_state_variables(v, phi, projection_field)
            else:
                dvs = []
                dpsis = []
                projection_field_orig = np.max(projection_field, axis=-1)
                dv_orig, dpsi_orig = statevarcomp.compute_state_variables(v, phi, projection_field_orig)
                for i in range(projection_field.shape[-1]):
                    dvi, dpsii = statevarcomp.compute_state_variables(v, phi, projection_field[:, i])
                    dvs.append(dvi)
                    dpsis.append(dpsii)
                dv = np.sum(dvs)
                dpsi = np.sum(dpsis)
                # print(f"According to original algorithm: dv={dv_orig}, dpsi={dpsi_orig}")
                # print(dvs, dpsis)
                # print(f"With Improved edge overlay (mean): dv={np.mean(dvs)}, dpsi={np.mean(dpsis)}")
                # print(f"With Improved edge overlay (sum): dv={np.sum(dvs)}, dpsi={np.sum(dpsis)}")

            if v > 0:
                v = min(v, 400)
            elif v < 0:
                v = max(v, -400)

            v = float(v)
            dpsi = float(dpsi)

            ## TODO: this is temporary smooth reandom walk
            if np.mean(projection_field) == 0 and control.SMOOTH_RW:
                if rw_dt > 2:
                    new_dpsi = np.random.uniform(-add_psi, add_psi, 1)
                    rw_dt = 0
                    # the more time spent without social cues the more extensive the exploration is
                    if add_psi < 1.5:
                        logger.error(f'add dpsi, {add_psi}')
                        add_psi += 0.1
                dpsi = new_dpsi
                rw_dt += dt
            else:
                # logger.error('zerodpsi')
                add_psi = 0.1

            # Initializing control parameters
            if is_initialized:
                v += dv * dt
                dpsi

            else:
                is_initialized = True
                dv = float(0)
                dpsi = float(0)

            # now_sign = np.sign(dv)

            # prev_sign = now_sign
            t_prev = t_now

            if monitoring.SAVE_CONTROL_PARAMS and not simulation.ENABLE_SIMULATION:

                # take a timestamp for this measurement
                time = datetime.datetime.utcnow()

                # generating data to dump in db
                field_dict = {"agent_velocity": dv,
                              "heading_angle": dpsi,
                              "processing_delay": (time - capture_timestamp).total_seconds()}

                # format the data as a single measurement for influx
                body = [
                    {
                        "measurement": measurement_name,
                        "time": time,
                        "fields": field_dict
                    }
                ]

                ifclient.write_points(body, time_precision='ms')

            if with_control:
                #ating excitation of left vs right of the projection field
                exc_left = np.mean(projection_field[0:int(len(projection_field) / 2)])
                exc_right = np.mean(projection_field[int(len(projection_field) / 2):])
                control_stream.put((v, dpsi, exc_left, exc_right))
                motor_control_mode_stream.put(movement_mode)

            if monitoring.ENABLE_CLOUD_STORAGE:
                with open(statevars_fpath, 'ab') as sv_f:
                    statevars = np.concatenate((np.array([t_now]),
                                                np.array([dv, dpsi])))
                    pickle.dump(statevars, sv_f)

            # To test infinite loops
            if env.EXIT_CONDITION:
                break

    except KeyboardInterrupt:
        pass
