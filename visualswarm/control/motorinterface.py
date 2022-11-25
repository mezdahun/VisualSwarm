import os
import logging

from visualswarm.contrib import simulation

if not simulation.ENABLE_SIMULATION:
    from dbus.exceptions import DBusException

import time
from visualswarm.contrib import control, logparams

# using main logger
# setup logging
ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
bcolors = logparams.BColors


def asebamedulla_health(network):
    """Checking health of the established connection by requesting robot health"""
    logger.info(f'{bcolors.OKBLUE}HEALTHCHECK{bcolors.ENDC} asebamedulla connection')
    # Check Thymio's health
    try:
        network.GetVariable("thymio-II", "acc", timeout=5)
        return True
    except DBusException:
        return False


def asebamedulla_init():
    """Establishing initial connection with the Thymio robot on a predefined interface
        Args: None
        Vars: visualswarm.control.THYMIO_DEVICE_PORT: serial port on which the robot is available for the Pi
        Returns: None
    """
    logger.info(f'{bcolors.OKBLUE}CONNECT{bcolors.ENDC} via asebamedulla on {control.THYMIO_DEVICE_PORT}')
    # TODO: fix this to be changable from config between wireless and wired connection
    # important to improve code: http://wiki.thymio.org/en:asebaswitchremap
    # the cable matters!!!
    # os.system(f"(asebamedulla ser:device={control.THYMIO_DEVICE_PORT} &)")  # nosec
    os.system("(asebamedulla ser:name=Thymio-II &)")  # nosec
    time.sleep(5)


def asebamedulla_end():
    """Killing all established asebamedulla processes"""
    logger.info(f'{bcolors.OKBLUE}CLOSE{bcolors.ENDC} connection via asebamedulla')
    os.system("pkill -f asebamedulla")  # nosec
    logger.info(f'{bcolors.OKGREEN}✓ CLOSE CONNECTION SUCCESSFUL{bcolors.ENDC} via asebamedulla')
