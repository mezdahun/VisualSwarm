import dbus
import dbus.mainloop.glib
import logging

from visualswarm.control import motorinterface
from visualswarm.contrib import logparams, control
from visualswarm import env

import numpy as np
import tempfile
from datetime import datetime
# import random
from time import sleep


sleep(10)
# using main logger
logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors

# # Initializing DBus
# bus = None
#
# # Create Aseba network
# network = None

def light_up_led(network, R, G, B):
    """
    Method to indicate movement mode by lighting up top LEDS on robot
        Args:
            network: DBUS network to reach Thymio2
            R, G, B: color configuration of led, min: (0, 0, 0), max: (32, 32, 32)
        Returns:
            None
    """
    with tempfile.NamedTemporaryFile(suffix='.aesl', mode='w+t') as aesl:
        aesl.write('<!DOCTYPE aesl-source>\n<network>\n')
        node_id = 1
        name = 'thymio-II'
        aesl.write(f'<node nodeId="{node_id}" name="{name}">\n')
        aesl.write(f'call leds.top({R},{G},{B})\n')
        aesl.write('</node>\n')
        aesl.write('</network>\n')
        aesl.seek(0)
        network.LoadScripts(aesl.name)


def step_random_walk() -> list:
    """
    Method to get motor velocity values according to a preconfigured random walk (RW) process
        Args:
            No args, configured via contrib.control
        Returns:
            [v_left, v_right]: RW motor values
    """
    # drawing random change in heading angle from uniform dist
    dpsi = np.random.uniform(-control.DPSI_MAX_EXP, control.DPSI_MAX_EXP, 1)

    # distributing desired forward velocity according to drwan change in h.a.
    [v_left, v_right] = distribute_overall_speed(control.V_EXP_RW, dpsi)

    return [v_left, v_right]


def rotate() -> list:
    """
    Method to get motor velocity values according to a preconfigured rotation (ROT) process
        Args:
            No args, configured via contrib.control
        Returns:
            [v_left, v_right]: ROT motor values
    """
    if control.ROT_DIRECTION == 'Left':
        right_sign = 1
    elif control.ROT_DIRECTION == 'Right':
        right_sign = -1
    elif control.ROT_DIRECTION == 'Random':
        right_sign = np.random.choice([1, -1], 1)[0]
    else:
        logger.error(f"Wrong configuration value control.ROT_DIRECTION=\"{control.ROT_DIRECTION}\"! Abort!")
        raise KeyboardInterrupt
    left_sign = -1 * right_sign

    v_left = left_sign * control.ROT_MOTOR_SPEED
    v_right = right_sign * control.ROT_MOTOR_SPEED
    return [v_left, v_right]


def hardlimit_motor_speed(v_left: float, v_right: float) -> list:
    """
    Process to limit the motor speed into the available physical domain of the robot.
        Args:
            v_left (float): motor velocity to hard limit on left engine
            v_right (float): motor velocity to hard limit on right engine
        Returns:
            [v_left_lim, v_right_lim]: limited motor values
    """
    # preserving signs
    if v_left != 0 and v_right != 0:
        sign_v_left = np.sign(v_left)
        sign_v_right = np.sign(v_right)

        if np.abs(v_left) > np.abs(v_right):
            v_right_prop = np.abs(v_right) / np.abs(v_left)
            v_left_lim = sign_v_left * control.MAX_MOTOR_SPEED
            v_right_lim = sign_v_right * v_left_lim * v_right_prop

        elif np.abs(v_left) == np.abs(v_right):
            v_right_lim = sign_v_right * control.MAX_MOTOR_SPEED
            v_left_lim = sign_v_left * control.MAX_MOTOR_SPEED

        else:
            v_left_prop = np.abs(v_left) / np.abs(v_right)
            v_right_lim = sign_v_right * control.MAX_MOTOR_SPEED
            v_left_lim = sign_v_left * v_right_lim * v_left_prop
    else:
        if v_left == 0:
            v_left_lim, v_right_lim = 0, np.sign(v_right) * control.MAX_MOTOR_SPEED
        else:
            v_left_lim, v_right_lim = np.sign(v_left) * control.MAX_MOTOR_SPEED, 0

    return [v_left_lim, v_right_lim]


def distribute_overall_speed(v: float, dpsi: float) -> list:
    """
    distributing desired forward speed to motor velocities according to the change in the heading angle dpsi.
        Args:
            v (float): desired forward speed of the agent
            dpsi (float): change in the heading angle of the agent
        Returns:
            [v_left, v_right]: motor velocity values of the agent
    """
    # Matching simulation scale with reality
    v = v * control.MOTOR_SCALE_CORRECTION

    # Calculating proportional heading angle change
    dpsi_p = dpsi / np.pi

    # Distributing velocity
    v_left = v * (1 + dpsi_p)
    v_right = v * (1 - dpsi_p)

    return [v_left, v_right]


def prox_emergency_callback(*args):
    logger.warning('EMERGENCY-EMERGENCY-EMERGENCY!!!!')


def first_function():
    logger.info("first")

def second_function():
    logger.info("second")

def control_thymio(control_stream, motor_control_mode_stream, with_control=False):
    """
    Process to translate state variables to motor velocities and send to Thymio2 robot via DBUS.
        Args:
            control_stream (multiprocessing Queue): stream to push calculated control parameters
            motor_control_mode_stream (multiprocessing Queue): stream to get movement type/mode.
            with_control (boolean): sends motor command to robot if true. Only consumes input stream if false.
        Returns:
            -shall not return-
    """
    # global network
    # global bus
    prev_movement_mode = "BEHAVE"
    (expR, expG, expB) = control.EXPLORE_STATUS_RGB
    (behR, behG, behB) = control.BEHAVE_STATUS_RGB

    if not with_control:
        # simply consuming the input stream so that we don't fill up memory
        while True:
            (v, dpsi) = control_stream.get()
            movement_mode = motor_control_mode_stream.get()

            # To test infinite loops
            if env.EXIT_CONDITION:
                break
    else:
        # Initialize timestamp
        last_explore_change = datetime.now()
        last_behave_change = datetime.now()

        # Initializing DBus
        dbus.mainloop.glib.threads_init()
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()

        # Create Aseba network
        # if network is None:
        network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                 dbus_interface='ch.epfl.mobots.AsebaNetwork')

        if motorinterface.asebamedulla_health(network):
            logger.info(f'{bcolors.OKGREEN}✓ CONNECTION SUCCESSFUl{bcolors.ENDC} via asebamedulla')

            while True:
                # fetching state variables
                (v, dpsi) = control_stream.get()
                movement_mode = motor_control_mode_stream.get()

                if movement_mode == "BEHAVE":

                    # Switch between modes, change mode status LED
                    if prev_movement_mode == "EXPLORE":
                        light_up_led(network, behR, behG, behB)

                    # Persistent change in movement mode
                    is_persistent = abs((last_explore_change - datetime.now()).total_seconds()) > \
                                    control.WAIT_BEFORE_SWITCH_MOVEMENT
                    if is_persistent or env.EXIT_CONDITION:
                        # Behavior according to Romanczuk and Bastien 2020
                        # distributing desired forward speed according to dpsi
                        [v_left, v_right] = distribute_overall_speed(v, dpsi)

                        # hard limit motor velocities but keep their ratio for desired movement
                        if np.abs(v_left) > control.MAX_MOTOR_SPEED or np.abs(v_right) > control.MAX_MOTOR_SPEED:
                            logger.warning(f'Reached max velocity: left:{v_left:.2f} right:{v_right:.2f}')
                            [v_left, v_right] = hardlimit_motor_speed(v_left, v_right)

                        # sending motor values to robot
                        network.SetVariable("thymio-II", "motor.left.target", [v_left])
                        network.SetVariable("thymio-II", "motor.right.target", [v_right])

                        logger.debug(f"BEHAVE left: {v_left} \t right: {v_right}")
                        # last time we changed velocity according to BEHAVIOR REGIME
                        last_behave_change = datetime.now()

                elif movement_mode == "EXPLORE":

                    # Switch between modes, change mode status LED
                    if prev_movement_mode == "BEHAVE":
                        light_up_led(network, expR, expG, expB)

                    # Persistent change in modes
                    if abs((last_behave_change - datetime.now()).total_seconds()) > control.WAIT_BEFORE_SWITCH_MOVEMENT:
                        # Enforcing specific dt in Random Walk Process
                        if abs((last_explore_change - datetime.now()).total_seconds()) > control.RW_DT:

                            if control.EXP_MOVE_TYPE == 'RandomWalk':
                                # Exploration according to Random Walk Process
                                [v_left, v_right] = step_random_walk()
                            elif control.EXP_MOVE_TYPE == 'Rotation':
                                # Exploration according to simple rotation movement
                                [v_left, v_right] = rotate()
                            else:
                                # Unknown exploration regime in configuration
                                logger.error(f"Unknown exploration type \"{control.EXP_MOVE_TYPE}\"! Abort!")
                                raise KeyboardInterrupt

                            logger.debug(f'EXPLORE left: {v_left} \t right: {v_right}')

                            # sending motor values to robot
                            network.SetVariable("thymio-II", "motor.left.target", [v_left])
                            network.SetVariable("thymio-II", "motor.right.target", [v_right])

                            # last time we changed velocity according to EXPLORE REGIME
                            last_explore_change = datetime.now()

                else:
                    logger.error(f"Unknown movement type \"{movement_mode}\"! Abort!")
                    raise KeyboardInterrupt

                prev_movement_mode = movement_mode

                # To test infinite loops
                if env.EXIT_CONDITION:
                    break
        else:
            logger.error(f'{bcolors.FAIL}🗴 CONNECTION FAILED{bcolors.ENDC} via asebamedulla')
            motorinterface.asebamedulla_end()
            raise Exception('asebamedulla connection not healthy!')

def emergency_behavior():
    # Initializing DBus
    dbus.mainloop.glib.threads_init()
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    t = datetime.now()

    # Create Aseba network
    # if network is None:
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                             dbus_interface='ch.epfl.mobots.AsebaNetwork')
    while True:
        if abs(t-datetime.now()).total_seconds() > 0.2:
            prox_val = np.array([val for val in network.GetVariable("thymio-II", "prox.horizontal")])
            idx = np.where(prox_val>2000)
            if len(idx) > 0:
                logger.info(idx)
                logger.info('EMERGENCY')
                logger.info(prox_val)
            t =datetime.now()

    # from gi.repository import GLib
    # loop = GLib.MainLoop()
    #
    #
    # # Initializing DBus
    # dbus.mainloop.glib.threads_init()
    # dbus.mainloop.glib.DBusGMainLoop(mainloop=loop)
    # bus = dbus.SessionBus()
    #
    # # Create Aseba network
    # # if network is None:
    # network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
    #                          dbus_interface='ch.epfl.mobots.AsebaNetwork')
    #
    # with tempfile.NamedTemporaryFile(suffix='.aesl', mode='w+t', delete=False) as aesl:
    #     aesl.write('<!DOCTYPE aesl-source>\n<network>\n')
    #     # declare global events and ...
    #     aesl.write('<event size="0" name="fwd.button.backward"/>\n')
    #     aesl.write('<event size="0" name="become.yellow"/>\n')
    #     aesl.write('<event size="0" name="fwd.timer0"/>\n')
    #     thymio = "thymio-II"
    #     aesl.write('<node nodeId="1" name="' + thymio + '">\n')
    #     # ...forward some local events as outgoing global ones
    #     aesl.write('onevent button.forward\n    emit fwd.button.backward\n')
    #     aesl.write('onevent timer0\n    emit fwd.timer0\n')
    #     # add code to handle incoming events
    #     aesl.write('onevent become.yellow\n    call leds.top(31,31,0)\n')
    #     aesl.write('</node>\n')
    #     aesl.write('</network>\n')
    #     aesl.seek(0)
    #
    # network.LoadScripts(aesl.name)
    #
    # # # Create an event filter and catch events
    # # eventfilter = network.CreateEventFilter()
    # # events = dbus.Interface(
    # #     bus.get_object('ch.epfl.mobots.Aseba', eventfilter),
    # #     dbus_interface='ch.epfl.mobots.EventFilter')
    # #
    # # network.SetVariable(thymio, "timer.period", [1000, 0])
    # # events.ListenEventName('fwd.timer0')  # not required for the first event in aesl file!
    # # events.ListenEventName('fwd.button.backward')
    # # events.connect_to_signal('Event', prox_emergency_callback)
    #
    # while True:
    #     # Create an event filter and catch events
    #     eventfilter = network.CreateEventFilter()
    #     events = dbus.Interface(
    #         bus.get_object('ch.epfl.mobots.Aseba', eventfilter),
    #         dbus_interface='ch.epfl.mobots.EventFilter')
    #
    #     network.SetVariable(thymio, "timer.period", [1000, 0])
    #     events.ListenEventName('fwd.timer0')  # not required for the first event in aesl file!
    #     events.ListenEventName('fwd.button.backward')
    #     events.connect_to_signal('Event', prox_emergency_callback)
