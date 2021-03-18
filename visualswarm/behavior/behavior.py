"""
@author: mezdahun
@description: Submodule to implement control according to visual stream
"""
import datetime
import logging
from math import floor

import numpy as np

import visualswarm.contrib.vision
from visualswarm.monitoring import ifdb
from visualswarm.contrib import monitoring
from visualswarm.behavior import statevarcomp
from visualswarm import env

# using main logger
logger = logging.getLogger('visualswarm.app')


def VPF_to_behavior(VPF_stream, control_stream, with_control=False):
    """
    Process to extract final visual projection field from high level visual input.
        Args:
            VPF_stream (multiprocessing Queue): stream to receive visual projection field
            control_stream (multiprocessing Queue): stream to push calculated control parameters
        Returns:
            -shall not return-
            :param with_control:
    """
    measurement_name = "control_parameters"
    ifclient = ifdb.create_ifclient()
    phi = None
    v = 0
    psi = 0

    (projection_field, capture_timestamp) = VPF_stream.get()
    phi = np.linspace(visualswarm.contrib.vision.PHI_START, visualswarm.contrib.vision.PHI_END,
                      len(projection_field))

    # calculating theoretically max value for velocity change for normalization
    max_VPF = np.zeros(len(projection_field))
    max_VPF[floor(len(projection_field) / 2)] = 1
    dv_max, dpsi_max = statevarcomp.compute_state_variables(v, phi, max_VPF)
    logger.info(f"Theoretical MAX: {dv_max}, {dpsi_max}")

    while True:
        (projection_field, capture_timestamp) = VPF_stream.get()

        dv, dpsi = statevarcomp.compute_state_variables(v, phi, projection_field)
        v += dv
        psi += dpsi
        psi = psi % (2 * np.pi)

        if monitoring.SAVE_CONTROL_PARAMS:

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

        # To test infinite loops
        if env.EXIT_CONDITION:
            break
