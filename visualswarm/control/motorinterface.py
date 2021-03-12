import os
import dbus
import dbus.mainloop.glib
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
    global is_robot_healthy

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    # Create Aseba network
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                             dbus_interface='ch.epfl.mobots.AsebaNetwork')

    # Check Thymio's health
    test_var = None
    test_var = network.GetVariable("thymio-II", "acc", timeout=5)

    if test_var is not None:
        return True
    else:
        return False


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
