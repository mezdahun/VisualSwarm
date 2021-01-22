"""
@author: mezdahun
@description: Submodule to implement control according to visual stream
"""
import datetime
import logging

import numpy as np

from visualswarm.monitoring import ifdb
from visualswarm.contrib import projection, monitorparams, flockparams
from visualswarm.behavior import movecomp
from visualswarm import env

# using main logger
logger = logging.getLogger('visualswarm.app')


def VPF_to_behavior(VPF_stream, control_stream):
    """
    Process to extract final visual projection field from high level visual input.
        Args:
            VPF_stream: stream to receive visual projection field
            control_stream: stream to push calculated control parameters
        Returns:
            -shall not return-
    """
    measurement_name = "control_parameters"
    ifclient = ifdb.create_ifclient()
    phi = None
    v = 0
    psi = 0

    while True:
        (projection_field, capture_timestamp) = VPF_stream.get()
        if phi is None:
            phi = np.linspace(projection.PHI_START, projection.PHI_END, len(projection_field))

        dv, dpsi = movecomp.compute_control_params(v, phi, projection_field)
        v += dv
        psi += dpsi
        if np.abs(v) > flockparams.V_MAX_PHYS:
            if v > 0:
                v = float(flockparams.V_MAX_PHYS)
            else:
                v = -float(flockparams.V_MAX_PHYS)
        if psi > flockparams.DPSI_MAX_PHYS:
            if psi > 0:
                psi = float(flockparams.DPSI_MAX_PHYS)
            else:
                psi = -float(flockparams.DPSI_MAX_PHYS)

        if monitorparams.SAVE_CONTROL_PARAMS:

            # take a timestamp for this measurement
            time = datetime.datetime.utcnow()

            # generating data to dump in db
            field_dict = {"agent_velocity": v,
                          "heading_angle": psi,
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

        # To test infinite loops
        if env.EXIT_CONDITION:
            break
