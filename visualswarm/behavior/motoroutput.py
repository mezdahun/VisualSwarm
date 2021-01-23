import dbus
import dbus.mainloop.glib
from gi.repository import GObject as gobject
from gi.repository import GLib
from optparse import OptionParser
import time
from multiprocessing import Process, Queue
import tempfile
import random
import numpy as np

# Create a global variable or Queue for GetVariable values
# to get and store Thymio sensor values
proxSensorsVal = [0, 0, 0, 0, 0]


def test_motor_control():
    # get the values of the sensors
    network.GetVariable("thymio-II", "prox.horizontal", reply_handler=handle_GetVariable_reply,
                        error_handler=handle_GetVariable_error)

    # print the proximity sensors value in the terminal
    print(proxSensorsVal[0], proxSensorsVal[1], proxSensorsVal[2], proxSensorsVal[3], proxSensorsVal[4])

    with tempfile.NamedTemporaryFile(suffix='.aesl', mode='w+t') as aesl:
        aesl.write('<!DOCTYPE aesl-source>\n<network>\n')
        node_id = 1
        name = 'thymio-II'
        aesl.write(f'<node nodeId="{node_id}" name="{name}">\n')
        # add code to handle incoming events
        R = random.randint(0, 32)
        G = random.randint(0, 32)
        B = random.randint(0, 32)
        aesl.write(f'call leds.top({R},{G},{B})\n')
        aesl.write('</node>\n')
        aesl.write('</network>\n')
        aesl.seek(0)
        network.LoadScripts(aesl.name)
    #
    # # Parameters of the Braitenberg, to give weight to each wheels
    # leftWheel = [-0.01, -0.005, -0.0001, 0.006, 0.015]
    # rightWheel = [0.012, +0.007, -0.0002, -0.0055, -0.011]
    #
    # # Braitenberg algorithm
    # totalLeft = 0
    # totalRight = 0
    # for i in range(5):
    #     totalLeft = totalLeft + (proxSensorsVal[i] * leftWheel[i])
    #     totalRight = totalRight + (proxSensorsVal[i] * rightWheel[i])
    #
    # # add a constant speed to each wheels so the robot moves always forward
    # totalRight = totalRight + 50
    # totalLeft = totalLeft + 50
    #
    # # print in terminal the values that is sent to each motor
    # print("totalLeft")
    # print(totalLeft)
    # print("totalRight")
    # print(totalRight)
    #
    # # send motor value to the robot
    # network.SetVariable("thymio-II", "motor.left.target", [totalLeft])
    # network.SetVariable("thymio-II", "motor.right.target", [totalRight])
    #
    return True


def control_thymio(control_stream):
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    # Create Aseba network
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                             dbus_interface='ch.epfl.mobots.AsebaNetwork')
    while True:
        (v, psi) = control_stream.get()
        # print('-------------------')
        print(psi)
        # psi -= np.pi
        # print(psi)
        psi = psi/2 # np.pi
        # print(psi)
        v_left = v * (1 + psi)/2 * 100
        v_right = v * (1 - psi)/2 * 100
        # print(v_left, v_right)
        network.SetVariable("thymio-II", "motor.left.target", [v_left])
        network.SetVariable("thymio-II", "motor.right.target", [v_right])


def handle_GetVariable_reply(r):
    global proxSensorsVal
    proxSensorsVal = r


def handle_GetVariable_error(e):
    raise Exception(str(e))


def execute_motor_control_test():
    # print in the terminal the name of each Aseba Node
    print(network.GetNodesList())

    # GObject loop
    loop = GLib.MainLoop()
    # call the callback of test_motor_control in every iteration
    GLib.timeout_add(100, test_motor_control)  # every 0.1 sec
    loop.run()


def execute_control_thymio(control_stream, network, loop):
    # print in the terminal the name of each Aseba Node
    # gobject.threads_init()
    # print(network.GetNodesList())

    # # GObject loop
    # loop = GLib.MainLoop()
    # call the callback of test_motor_control in every iteration
    GLib.timeout_add(100, control_thymio, control_stream, network)  # every 0.1 sec
    loop.run()


# if __name__ == '__main__':
#     gobject.threads_init()
#     dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
#
#     # if options.system:
#     #     bus = dbus.SystemBus()
#     # else:
#     bus = dbus.SessionBus()
#
#     # Create Aseba network
#     network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
#                              dbus_interface='ch.epfl.mobots.AsebaNetwork')
#     motor_control = Process(target=execute_motor_control_test)
#     motor_control.start()
