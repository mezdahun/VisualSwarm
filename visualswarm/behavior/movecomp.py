"""
@author: mezdahun
@description: Submodule to implement main behavioral/movement computations defined by
https://advances.sciencemag.org/content/6/6/eaay0792
"""
import logging

import numpy as np
from scipy import integrate

from visualswarm.contrib import flockparams

# using main logger
logger = logging.getLogger('visualswarm.app')


def dPhi_V_of(Phi, V):
    """Calculating derivative of VPF according to Phi visual angle at a given timepoint t"""
    # circular padding for edge cases
    padV = np.pad(V, (1, 1), 'wrap')
    dPhi_V_raw = np.diff(padV)
    # print(f'before: {np.count_nonzero(dPhi_V_raw)}')
    # we want to include non-zero value if it is on the edge
    if dPhi_V_raw[0] > 0 and dPhi_V_raw[-1] > 0:
        # print('edge case')
        dPhi_V_raw = dPhi_V_raw[0:-1]
    else:
        dPhi_V_raw = dPhi_V_raw[1:, ...]
    dPhi_V = dPhi_V_raw / (Phi[-1]-Phi[-2])
    return dPhi_V


def dt_V_of(t, joined_V):
    """Calculating the temporal derivative of VPF to all Phi visual angles"""
    dt_V = np.diff(joined_V, axis=0, prepend=0) / np.diff(t, prepend=0)
    return dt_V


def compute_control_params(vel_now, phi, V_now, t_now=None, V_prev=None, t_prev=None):
    """Calculating the velocity difference of the agent according the main algorithm"""
    # Deriving over t
    if V_prev is not None or t_prev is not None or t_now is not None:
        logger.debug('Movement calculation called with NONE as time-related parameters.')
        t_vec = np.hstack((t_prev, t_now))
        joined_V = np.vstack((V_prev, t_prev))
        dt_V = dt_V_of(t_vec, joined_V)
    else:
        dt_V = np.zeros(len(phi))

    # Deriving over Phi
    dPhi_V = dPhi_V_of(phi, V_now)

    # Calculating series expansion of functional G
    G_vel = flockparams.ALP0 * (-V_now + flockparams.ALP2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_vel_spike = flockparams.ALP0 * flockparams.ALP1 * np.square(dPhi_V)

    G_psi = flockparams.BET0 * (-V_now + flockparams.BET2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_psi_spike = flockparams.BET0 * flockparams.BET1 * np.square(dPhi_V)

    # Calculating change in velocity and heading direction
    dphi = phi[-1] - phi[-2]
    spikey_part = np.sum(flockparams.ALP0 * flockparams.ALP1 * np.square(dPhi_V)) * dphi
    dvel = flockparams.GAM * (flockparams.V0 - vel_now) + \
           integrate.trapz(np.cos(phi) * G_vel, phi) + \
           np.sum(np.cos(phi) * G_vel_spike) * dphi
    dpsi = integrate.trapz(np.sin(phi) * G_psi, phi) + \
           np.sum(np.sin(phi) * G_psi_spike) * dphi
    return dvel, dpsi
