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

# using main logger
logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors


def light_up_led(network, R, G, B):

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
    return True


def step_random_walk() -> list:
    """
    Method to get motor velocity values according to a preconfigured random walk (RW) process
        Args:
            No args, configured via contrib.control
        Returns:
            [v_left_lim, v_right_lim]: RW motor values
    """
    dpsi = np.random.uniform(-control.DPSI_MAX_EXP, control.DPSI_MAX_EXP, 1)
    [v_left, v_right] = distribute_overall_speed(control.V_EXP_RW, dpsi)
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
    v_left = v * (1 - dpsi_p)
    v_right = v * (1 + dpsi_p)

    return [v_left, v_right]


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
                            # Exploration according to Random Walk Process
                            [v_left, v_right] = step_random_walk()
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
