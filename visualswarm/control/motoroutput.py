import dbus
import dbus.mainloop.glib
import logging
from numpy import sign

from visualswarm.control import motorinterface
from visualswarm.contrib import logparams

import tempfile
import random

# Create a global variable or Queue for GetVariable values
# to get and store Thymio sensor values
proxSensorsVal = [0, 0, 0, 0, 0]

# using main logger
logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors


def handle_GetVariable_reply(r):
    global proxSensorsVal
    proxSensorsVal = r


def handle_GetVariable_error(e):
    raise Exception(str(e))


def test_motor_control(network):
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
        R = random.randint(0, 32)  # nosec
        G = random.randint(0, 32)  # nosec
        B = random.randint(0, 32)  # nosec
        aesl.write(f'call leds.top({R},{G},{B})\n')
        aesl.write('</node>\n')
        aesl.write('</network>\n')
        aesl.seek(0)
        network.LoadScripts(aesl.name)
    return True


def control_thymio(control_stream, with_control=False):
    if not with_control:
        # simply consuming the input stream so that we don't fill up memory
        while True:
            (v, psi) = control_stream.get()
    else:
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()

        # Create Aseba network
        network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                 dbus_interface='ch.epfl.mobots.AsebaNetwork')
        if motorinterface.asebamedulla_health(network):
            logger.info(f'{bcolors.OKGREEN}✓ CONNECTION SUCCESSFUl{bcolors.ENDC} via asebamedulla')
            while True:
                v_max_motor = 500
                (v, dpsi) = control_stream.get()

                # v_left_current = network.GetVariable("thymio-II", "motor.left.target")
                # v_right_current = network.GetVariable("thymio-II", "motor.right.target")

                v_left = v * (1 + dpsi) / 2 * 100
                v_right = v * (1 - dpsi) / 2 * 100

                # v_left_perc = v_left / (abs(v_left) + abs(v_right))
                # v_right_perc = v_right / (abs(v_left) + abs(v_right))
                #
                # try:
                #     v_left = int(v_left_perc*v_max_motor)
                #     v_right = int(v_right_perc*v_max_motor)
                # except ValueError:
                #     v_left = 0
                #     v_right = 0

                # v_left_change = dv_norm * v_max_motor * (1 + dpsi) / 2
                # v_left = v_left_current + v_left_change
                # if abs(v_left) >= v_max_motor:
                #     v_left = sign(v_left) * v_max_motor
                #
                # v_right_change = dv_norm * v_max_motor * (1 - dpsi) / 2
                # v_right = v_right_current + v_right_change
                # if abs(v_right) >= v_max_motor:
                #     v_right = sign(v_right) * v_max_motor

                network.SetVariable("thymio-II", "motor.left.target", [v_left])
                network.SetVariable("thymio-II", "motor.right.target", [v_right])
                logger.info(f"left: {v_left} \t right: {v_right}")
        else:
            logger.error(f'{bcolors.FAIL}🗴 CONNECTION FAILED{bcolors.ENDC} via asebamedulla')
            motorinterface.asebamedulla_end()
            raise Exception('asebamedulla connection not healthy!')
