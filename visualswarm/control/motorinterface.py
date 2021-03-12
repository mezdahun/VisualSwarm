import os
import dbus
import dbus.mainloop.glib
from dbus.exceptions import DBusException
import time
from visualswarm.contrib import control

is_robot_healthy = False


def healthy_response(v):
    global is_robot_healthy
    print("called healthy callback")
    is_robot_healthy = True


def unhealthy_response(e):
    global is_robot_healthy
    print("called unhealthy callback")
    is_robot_healthy = False


def asebamedulla_health():
    """Checking health of the established connection by requesting robot health"""
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

    # Check Thymio's health
    try:
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
    info = os.system(f"(asebamedulla ser:device={control.THYMIO_DEVICE_PORT} &)")
    time.sleep(5)
    print('checking for health')
    if not asebamedulla_health():
        raise Exception('Connection can not be established with robot!')
    else:
        print("Connection via asebamedulla is healthy!")


def asebamedulla_end():
    """Killing all established asebamedulla processes"""
    os.system("pkill -f asebamedulla")
