import dbus
import dbus.mainloop.glib
import logging

from visualswarm.control import motorinterface
from visualswarm.contrib import logparams, control

# import tempfile
# import random

# using main logger
logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors


# def test_motor_control(network):
#     # get the values of the sensors
#     proxSensorsVal = network.GetVariable("thymio-II", "prox.horizontal")
#
#     # print the proximity sensors value in the terminal
#     print(proxSensorsVal[0], proxSensorsVal[1], proxSensorsVal[2], proxSensorsVal[3], proxSensorsVal[4])
#
#     with tempfile.NamedTemporaryFile(suffix='.aesl', mode='w+t') as aesl:
#         aesl.write('<!DOCTYPE aesl-source>\n<network>\n')
#         node_id = 1
#         name = 'thymio-II'
#         aesl.write(f'<node nodeId="{node_id}" name="{name}">\n')
#         # add code to handle incoming events
#         R = random.randint(0, 32)  # nosec
#         G = random.randint(0, 32)  # nosec
#         B = random.randint(0, 32)  # nosec
#         aesl.write(f'call leds.top({R},{G},{B})\n')
#         aesl.write('</node>\n')
#         aesl.write('</network>\n')
#         aesl.seek(0)
#         network.LoadScripts(aesl.name)
#     return True


def control_thymio(control_stream, with_control=False):
    """
    Process to translate state variables to motor velocities and send to Thymio2 robot via DBUS.
        Args:
            control_stream (multiprocessing Queue): stream to push calculated control parameters
            with_control (boolean): sends motor command to robot if true. Only consumes input stream if false.
        Returns:
            -shall not return-
    """
    if not with_control:
        # simply consuming the input stream so that we don't fill up memory
        while True:
            (v, dpsi) = control_stream.get()
    else:
        # Initializing DBus
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()

        # Create Aseba network
        network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                 dbus_interface='ch.epfl.mobots.AsebaNetwork')

        if motorinterface.asebamedulla_health(network):
            logger.info(f'{bcolors.OKGREEN}✓ CONNECTION SUCCESSFUl{bcolors.ENDC} via asebamedulla')

            while True:
                # fetching state variables
                (v, dpsi) = control_stream.get()

                # distributing v according dpsi to the differential system
                v_left = v * (1 + dpsi) / 2 * control.MOTOR_SCALE_CORRECTION
                v_right = v * (1 - dpsi) / 2 * control.MOTOR_SCALE_CORRECTION

                # sending motor values to robot
                network.SetVariable("thymio-II", "motor.left.target", [v_left])
                network.SetVariable("thymio-II", "motor.right.target", [v_right])

                logger.info(f"left: {v_left} \t right: {v_right}")
        else:
            logger.error(f'{bcolors.FAIL}🗴 CONNECTION FAILED{bcolors.ENDC} via asebamedulla')
            motorinterface.asebamedulla_end()
            raise Exception('asebamedulla connection not healthy!')
