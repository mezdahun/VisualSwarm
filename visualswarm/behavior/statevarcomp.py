"""
@author: mezdahun
@description: Submodule to implement main behavioral/movement computations defined by
https://advances.sciencemag.org/content/6/6/eaay0792
"""
import logging

import numpy as np
import numpy.typing as npt
from scipy import integrate

from visualswarm.contrib import behavior, monitoring

# using main logger
# setup logging
import os

ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
logger.setLevel(monitoring.LOG_LEVEL)


def dPhi_V_of(Phi: npt.ArrayLike, V: npt.ArrayLike) -> npt.ArrayLike:
    """Calculating derivative of VPF according to Phi visual angle array at a given timepoint t
        Args:
            Phi: linspace numpy array of visual field axis
            V: binary visual projection field array
        Returns:
            dPhi_V: derivative array of V w.r.t Phi
    """
    # circular padding for edge cases
    padV = np.pad(V, (1, 1), 'wrap')
    dPhi_V_raw = np.diff(padV)

    # we want to include non-zero value if it is on the edge
    if dPhi_V_raw[0] > 0 and dPhi_V_raw[-1] > 0:
        dPhi_V_raw = dPhi_V_raw[0:-1]

    else:
        dPhi_V_raw = dPhi_V_raw[1:, ...]

    dPhi_V = dPhi_V_raw  # / (Phi[-1] - Phi[-2])
    return dPhi_V


# def dt_V_of(dt, joined_V):
#     """Calculating the temporal derivative of VPF to all Phi visual angles"""
#     dt_V = np.diff(joined_V, axis=0, prepend=0) / dt
#     return dt_V


def compute_state_variables(vel_now: float, Phi: npt.ArrayLike, V_now: npt.ArrayLike,
                            t_now=None, V_prev=None, t_prev=None,
                            GAM=None, V0=None,
                            ALP0=None, ALP1=None, ALP2=None,
                            BET0=None, BET1=None, BET2=None):
    """Calculating state variables of a given agent according to the main algorithm as in
    https://advances.sciencemag.org/content/6/6/eaay0792.
        Args:
            vel_now: current speed of the agent
            V_now: current binary visual projection field array
            Phi: linspace numpy array of visual field axis
            t_now: current time
            V_prev: previous binary visual projection field array
            t_prev: previous time
        Optional:
            All behavioral parameter can be optionally passed to the
            function if default values are to be overwritten, e.g.
            when multiple classes are detected
        Returns:
            dvel: temporal change in agent velocity
            dpsi: temporal change in agent heading angle

    """
    # # Deriving over t
    # if V_prev is not None and t_prev is not None and t_now is not None:
    #     dt = t_now - t_prev
    #     logger.debug('Movement calculation called with NONE as time-related parameters.')
    #     joined_V = np.vstack((V_prev, t_prev))
    #     dt_V = dt_V_of(dt, joined_V)
    # else:
    #     dt_V = np.zeros(len(Phi))

    # getting behavioral parameters if they are not overwritten
    if GAM is None:
        GAM = behavior.GAM
    if V0 is None:
        V0 = behavior.V0
    if ALP0 is None:
        ALP0 = behavior.ALP0
    if ALP1 is None:
        ALP1 = behavior.ALP1
    if ALP2 is None:
        ALP2 = behavior.ALP2
    if BET0 is None:
        BET0 = behavior.BET0
    if BET1 is None:
        BET1 = behavior.BET1
    if BET2 is None:
        BET2 = behavior.BET2

    dt_V = np.zeros(len(Phi))

    # Deriving over Phi
    dPhi_V = dPhi_V_of(Phi, V_now)

    # Calculating series expansion of functional G
    G_vel = (-V_now + ALP2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_vel_spike = np.square(dPhi_V)

    G_psi = (-V_now + BET2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_psi_spike = np.square(dPhi_V)

    # Calculating change in velocity and heading direction
    dPhi = Phi[-1] - Phi[-2]
    FOV_rescaling_cos = 1
    FOV_rescaling_sin = 1

    # dvel = GAM * (V0 - vel_now) + \
    #        ALP0 * integrate.trapz(np.cos(FOV_rescaling_cos * Phi) * G_vel, Phi) + \
    #        ALP0 * ALP1 * np.sum(np.cos(Phi) * G_vel_spike)  #* dPhi
    # dpsi = BET0 * integrate.trapz(np.sin(Phi) * G_psi, Phi) + \
    #        BET0 * BET1 * np.sum(np.sin(FOV_rescaling_sin * Phi) * G_psi_spike)  #* dPhi

    dvel = GAM * (V0 - vel_now) + \
           ALP0 * integrate.trapz(np.cos(FOV_rescaling_cos * Phi) * G_vel, Phi) + \
           ALP0 * ALP1 * np.sum(np.cos(Phi) * G_vel_spike)  #* dPhi
    dpsi = BET0 * integrate.trapz((sigmoid(Phi, s=3*np.pi)+sigmoid(Phi, s=np.pi)) * G_psi, Phi) + \
           BET0 * BET1 * np.sum((sigmoid(Phi, s=3*np.pi)+sigmoid(Phi, s=np.pi)) * G_psi_spike)  #* dPhi

    return dvel, dpsi


def sigmoid(x, s):
    return 2 / (1 + np.exp(-s*x)) - 1