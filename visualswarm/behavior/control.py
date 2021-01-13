"""
@author: mezdahun
@description: Submodule to implement control according to visual stream
"""
import logging

import numpy as np

from visualswarm.monitoring import ifdb
from visualswarm.contrib import projection, flockparams
from visualswarm.behavior import movecomp

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
    while True:
        projection_field = VPF_stream.get()
        if phi is None:
            phi = np.linspace(projection.PHI_START, projection.PHI_END, len(projection_field))

        dv = movecomp.comp_velocity(v, phi, projection_field)
        v += dv

        if flockparams.SAVE_CONTROL_PARAMS:
            pass
            # # Saving projection field data to InfluxDB to visualize with Grafana
            # proj_field_vis = projection_field[0:-1:projection.DOWNGRADING_FACTOR]
            #
            # # take a timestamp for this measurement
            # time = datetime.datetime.utcnow()
            #
            # # generating data to dump in db
            # keys = [f'{ifdb.pad_to_n_digits(i)}' for i in range(len(proj_field_vis))]
            # field_dict = dict(zip(keys, proj_field_vis))
            #
            # # format the data as a single measurement for influx
            # body = [
            #     {
            #         "measurement": measurement_name,
            #         "time": time,
            #         "fields": field_dict
            #     }
            # ]
            #
            # ifclient.write_points(body, time_precision='ms')
        pass
