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
from visualswarm.behavior import statevarcomp
from visualswarm import env

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

        (projection_field, capture_timestamp) = VPF_stream.get()
        phi = np.linspace(visualswarm.contrib.vision.PHI_START, visualswarm.contrib.vision.PHI_END,
                          len(projection_field))

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
            (projection_field, capture_timestamp) = VPF_stream.get()

            if np.mean(projection_field) == 0 and control.EXP_MOVE_TYPE != 'NoExploration':
                movement_mode = "EXPLORE"
            else:
                movement_mode = "BEHAVE"

            t_now = datetime.datetime.now()
            dt = (t_now - t_prev).total_seconds()  # to normalize

            ## TODO: Find out what causes weird turning behavior
            #v = 0 # only to measure equilibrium distance. set v0 to zero too
            dv, dpsi = statevarcomp.compute_state_variables(v, phi, projection_field)
            if dpsi > 0:
                dpsi = min(dpsi, 1)
            elif dpsi < 0:
                dpsi = max(dpsi, -1)
            # if dpsi_before is None:
            #     dpsi_before = dpsi
            # delta_dpsi = dpsi - dpsi_before
            # if delta_dpsi > 0.5:
            #     dpsi = dpsi_before + 0.5
            # elif delta_dpsi < -0.5:
            #     dpsi = dpsi_before - 0.5
            # print(f"DPSI: {dpsi}")

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

            if is_initialized:
                v += dv * dt
            else:
                is_initialized = True
                dv = float(0)
                dpsi = float(0)

            # now_sign = np.sign(dv)

            # logger.warning(f'dV={dv * dt} with passed seconds {(t_now - start_behave).total_seconds()}')
            # if np.abs(now_sign - prev_sign) == 2 and (t_now - start_behave).total_seconds() > 15:
            #     control_stream.put((0, 0))
            #     motor_control_mode_stream.put("BEHAVE")
            #     logger.warning('STOP EXPERIMENT!!!!')
            #     #raise KeyboardInterrupt('DV decreased to zero and already 5 sec gone from experiment!!!')
            #     return


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
                control_stream.put((v, dpsi))
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
