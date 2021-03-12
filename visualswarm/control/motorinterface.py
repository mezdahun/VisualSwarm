import os
import dbus
import dbus.mainloop.glib
import logging

from dbus.exceptions import DBusException
import time
from visualswarm.contrib import control, logparams

# using main logger
logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors


def asebamedulla_health():
    """Checking health of the established connection by requesting robot health"""
    logger.info(f'{bcolors.OKBLUE}HEALTHCHECK{bcolors.ENDC} asebamedulla connection')
    # Check Thymio's health
    try:
        # init the dbus main loop
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

        # get stub of the aseba network
        bus = dbus.SessionBus()
        asebaNetworkObject = bus.get_object('ch.epfl.mobots.Aseba', '/')

        # prepare interface
        asebaNetwork = dbus.Interface(
            asebaNetworkObject,
            dbus_interface='ch.epfl.mobots.AsebaNetwork'
        )

        asebaNetwork.GetVariable("thymio-II", "acc", timeout=5)
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
    info = os.system(f"(asebamedulla ser:device={control.THYMIO_DEVICE_PORT} &)")
    time.sleep(5)
    # if not asebamedulla_health():
    #     logger.error(f'{bcolors.FAIL}🗴 CONNECTION FAILED{bcolors.ENDC} via asebamedulla')
    #     asebamedulla_end()
    #     raise Exception('Connection could not be established with robot!')
    # else:
    #     logger.info(f'{bcolors.OKGREEN}✓ CONNECTION SUCCESSFUl{bcolors.ENDC} via asebamedulla')


def asebamedulla_end():
    """Killing all established asebamedulla processes"""
    logger.info(f'{bcolors.OKBLUE}CLOSE{bcolors.ENDC} connection via asebamedulla')
    os.system("pkill -f asebamedulla")
    logger.info(f'{bcolors.OKGREEN}✓ CLOSE CONNECTION SUCCESSFUL{bcolors.ENDC}via asebamedulla')
