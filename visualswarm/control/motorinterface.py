import os
import dbus
import dbus.mainloop.glib
from visualswarm.contrib import control


def asebamedulla_health():
    """Checking health of the established connection by requesting robot health"""
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    # Create Aseba network
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                             dbus_interface='ch.epfl.mobots.AsebaNetwork')

    # Check Thymio's health
    is_robot_healthy = network.GetVariable("thymio-II", "prox.horizontal", reply_handler=lambda v: True,
                                           error_handler=lambda e: False)

    print(is_robot_healthy)
    return is_robot_healthy


def asebamedulla_init():
    """Establishing initial connection with the Thymio robot on a predefined interface
        Args: None
        Vars: visualswarm.control.THYMIO_DEVICE_PORT: serial port on which the robot is available for the Pi
        Returns: None
    """
    os.system(f"(asebamedulla ser:device={control.THYMIO_DEVICE_PORT} &) && sleep 1")
    if not asebamedulla_health():
        raise Exception('Connection can not be established with robot!')


def asebamedulla_end():
    """Killing all established asebamedulla processes"""
    os.system("pkill -f asebamedulla")
