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

# using main logger
if not simulation.ENABLE_SIMULATION:
    # setup logging
    import os
    ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
    logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
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

        (projection_field, capture_timestamp) = VPF_stream.get()
        phi = np.linspace(visualswarm.contrib.vision.PHI_START, visualswarm.contrib.vision.PHI_END,
                          len(projection_field))

        while True:
            (projection_field, capture_timestamp) = VPF_stream.get()

            if np.mean(projection_field) == 0 and control.EXP_MOVE_TYPE != 'NoExploration':
                movement_mode = "EXPLORE"
            else:
                movement_mode = "BEHAVE"

            t_now = datetime.datetime.now()
            dt = (t_now - t_prev).total_seconds()  # to normalize

            dv, dpsi = statevarcomp.compute_state_variables(v, phi, projection_field)
            v += dv * dt

            t_prev = t_now

            if monitoring.SAVE_CONTROL_PARAMS and not simulation.ENABLE_SIMULATION:

                # take a timestamp for this measurement
                time = datetime.datetime.utcnow()

                # generating data to dump in db
                field_dict = {"agent_velocity": v,
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

            # To test infinite loops
            if env.EXIT_CONDITION:
                break

    except KeyboardInterrupt:
        pass
