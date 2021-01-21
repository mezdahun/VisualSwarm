import dbus
import dbus.mainloop.glib
from gi.repository import GObject as gobject
from gi.repository import GLib
from optparse import OptionParser
import time

proxSensorsVal = [0, 0, 0, 0, 0]


def Braitenberg():
    # get the values of the sensors
    network.GetVariable("thymio-II", "prox.horizontal", reply_handler=handle_GetVariable_reply,
                        error_handler=handle_GetVariable_error)

    # print the proximity sensors value in the terminal
    print(proxSensorsVal[0], proxSensorsVal[1], proxSensorsVal[2], proxSensorsVal[3], proxSensorsVal[4])
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


def handle_GetVariable_reply(r):
    global proxSensorsVal
    proxSensorsVal = r


def handle_GetVariable_error(e):
    raise Exception(str(e))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--system", action="store_true", dest="system", default=False,
                      help="use the system bus instead of the session bus")

    (options, args) = parser.parse_args()

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    if options.system:
        bus = dbus.SystemBus()
    else:
        bus = dbus.SessionBus()

    # Create Aseba network
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')

    # print in the terminal the name of each Aseba NOde
    print(network.GetNodesList())

    # for i in range(100):
    #     Braitenberg()
    #     time.sleep(0.5)

    # GObject loop
    print('starting loop')
    loop = GLib.MainLoop()  # gobject.MainLoop()
    # call the callback of Braitenberg algorithm
    handle = GLib.timeout_add(100, Braitenberg)  # gobject.timeout_add(100, Braitenberg)  # every 0.1 sec
    loop.run()
    print('after loop')
