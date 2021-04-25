import dbus
import dbus.mainloop.glib
import logging

from visualswarm.control import motorinterface
from visualswarm.contrib import logparams, control, physconstraints
from visualswarm import env

import numpy as np
import tempfile
from datetime import datetime
# import random
from time import sleep
from queue import Empty


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


def empty_queue(queue2empty):
    """
    emptying a FIFO Queue object from multiprocessing package as there is no explicit way to do this.
        Args:
            queue2empty (multiprocessing.Queue): queue object to be emptied
        Returns:
            status: True if successful
    """
    while not queue2empty.empty():
        try:
            queue2empty.get_nowait()
        except Empty:
            logger.debug('Emptied passed queue')
            return True
    logger.debug('Emptied passed queue')
    return True


def turn_robot(network, angle, emergency_stream, turning_motor_speed=50):
    """
    turning robot with a specified speed to a particular physical angle according to the heuristics (multipliers)
    defined in contrib.physconstraints
        Args:
            network (dbus network): network on which we communicate with Thymio
            angle (float or int): physical angle in degrees to turn the robot with
            emergency_stream (multiprocessing.Queue): stream to receive real time emergenecy status and proximity
                sensor values
            turning_motor_speed (int), optional: motor speed to turn the robot with
        Returns:
            None
        Note: recursively calling turn_avoid_obstacle if obstacle is detected during turning as well as this
            method is called from turn_avoid_obstacle. As a result the recursion is continued until the proximity
            sensors are free.
    """
    # TODO: implement turning with keeping forward speed that can be calculated from v_left and v_right as avg
    # TODO: add some small random noise to turning time so that we don't get stuck when objects symmetric? Check avoidnace for corners
    # translating desired physical values into motor velocities
    phys_turning_rate = turning_motor_speed * physconstraints.ROT_MULTIPLIER
    turning_time = np.abs(angle / phys_turning_rate)

    logger.debug(f"phys turning rate: {phys_turning_rate} deg/sec")
    logger.debug(f"turning time: {turning_time}")

    # emptying so far accumulated values from emergency stream before turning maneuver with monitoring
    # otherwise updating from a FIFO stream would cause delay in proximity values and action
    empty_queue(emergency_stream)

    # if we need to call the turning maneuver recursively
    recursive_obstacle = False

    # current proximity values to act according to
    proximity_values = None

    # continue until we reach the desired angle
    start_time = datetime.now()
    while abs(start_time - datetime.now()).total_seconds() < turning_time:

        # call obstacle avoidance recursively if we get emergency signal from emergency_stream
        if not recursive_obstacle:
            # the proximity sensors in this timestep are clear, we can just continue setting the turning motor speeds
            network.SetVariable("thymio-II", "motor.left.target", [np.sign(angle) * turning_motor_speed])
            network.SetVariable("thymio-II", "motor.right.target", [-np.sign(angle) * turning_motor_speed])

        else:
            # if the defined angle of turn was not enough to clear the proximity sensors we retry to recursively
            # call the turning maneuver according to the new proximity values. The frequency of this check is
            # restricted by the frequency of new elements in the emergency_stream, as the get method will wait for the
            # new element
            logger.debug('Recursive turning maneuver during obstacle detection...')
            turn_avoid_obstacle(network, proximity_values, emergency_stream)

        # update emergency status and proximity values from emergency stream with wait behavior (get).
        (recursive_obstacle, proximity_values) = emergency_stream.get()


def move_robot(network, direction, distance, emergency_stream, moving_motor_speed=50):
    """
    moving robot with a specified speed to a particular distance according to the heuristics (multipliers)
    defined in contrib.physconstraints
        Args:
            network (dbus network): network on which we communicate with Thymio
            direction (string): either "Forward" or "Backward"
            distance (float or int): physical distance in mm to move the robot with
            emergency_stream (multiprocessing.Queue): stream to receive real time emergenecy status and proximity
                sensor values
            moving_motor_speed (int), optional: motor speed to move the robot with
        Returns:
            None
        Note: recursively calling avoid_obstacle if obstacle is detected during moving as well as this
            method is called from avoid_obstacle after the turning maneuver. As a result the recursion is continued
            until the proximity sensors are free, after which all recursively called avoid_obstacle methods will return.
    """
    # TODO: implement moving with keeping forward speed that can be calculated from v_left and v_right as avg
    # Checking distance
    if distance < 0:
        logger.error(f'Negative distance in move_robot: {distance}'
                     f'PLease control the robot direction with the "direction" parameter instead of the distance sign!')
        raise KeyboardInterrupt

    # Checking direction
    if direction == "Forward":
        multiplier = physconstraints.FWD_MULTIPLIER
        movesign = 1
    elif direction == "Backward":
        multiplier = physconstraints.BWD_MULTIPLIER
        movesign = -1
    else:
        logger.error(f'Unknown direction: {direction}')
        raise KeyboardInterrupt

    # calculating motor values from desired physical values and heuristics
    phys_speed = moving_motor_speed * multiplier
    movement_time = distance / phys_speed

    logger.debug(f"phys speed: {phys_speed} mm/sec")
    logger.debug(f"movement time: {movement_time}")

    # from this point the method shows a lot of similarity with turn_robot, please check the comments there
    empty_queue(emergency_stream)

    recursive_obstacle = False
    proximity_values = None

    start_time = datetime.now()
    while abs(start_time - datetime.now()).total_seconds() < movement_time:

        if not recursive_obstacle:
            network.SetVariable("thymio-II", "motor.left.target", [movesign * moving_motor_speed])
            network.SetVariable("thymio-II", "motor.right.target", [movesign * moving_motor_speed])

        else:
            # TODO: check if we are locked and return if yes
            avoid_obstacle(network, proximity_values, emergency_stream)

        (recursive_obstacle, proximity_values) = emergency_stream.get()


def speed_up_robot(network, additional_motor_speed_multiplier, emergency_stream, protocol_time=0.5):
    """
    speeding up robot with a specified additional motor speed until back sensors are cleared.
        Args:
            network (dbus network): network on which we communicate with Thymio
            additional_motor_speed_multiplier (int): motor speed to multiply to current motor speeds with
            protocol_time (float): time in seconds to continue protocol with checking for proximity values.
            emergency_stream (multiprocessing.Queue): stream to receive real time emergenecy status and proximity
                sensor values
        Returns:
            None
        Note: recursively calling avoid_obstacle if obstacle is detected during moving as well as this
            method is called from avoid_obstacle after the turning maneuver. As a result the recursion is continued
            until the proximity sensors are free, after which all recursively called avoid_obstacle methods will return.
    """
    # first getting current motor values
    v_left_curr = network.GetVariable("thymio-II", "motor.left.speed")[0]
    v_right_curr = network.GetVariable("thymio-II", "motor.right.speed")[0]
    logger.info(v_left_curr)
    logger.info(v_right_curr)

    # v_left_target = np.sign(v_left_curr) * (np.abs(v_left_curr) + additional_motor_speed)
    # v_right_target = np.sign(v_right_curr) * (np.abs(v_right_curr) + additional_motor_speed)

    v_left_target = additional_motor_speed_multiplier * v_left_curr
    v_right_target = additional_motor_speed_multiplier * v_right_curr

    # from this point the method shows a lot of similarity with turn_robot, please check the comments there
    empty_queue(emergency_stream)

    recursive_obstacle = False
    proximity_values = None

    start_time = datetime.now()
    while abs(start_time - datetime.now()).total_seconds() < protocol_time:

        if not recursive_obstacle:
            network.SetVariable("thymio-II", "motor.left.target", [v_left_target])
            network.SetVariable("thymio-II", "motor.right.target", [v_right_target])

        else:
            # TODO: check if we are locked and return if yes
            avoid_obstacle(network, proximity_values, emergency_stream)

        (recursive_obstacle, proximity_values) = emergency_stream.get()


def turn_avoid_obstacle(network, prox_vals, emergency_stream, turn_avoid_angle=None):
    """
    deciding on and starting turning maneuver during obstacle avoidance.
        Args:
            network (dbus network): network on which we communicate with Thymio
            prox_vals (list or np.array): len 7 array with the proximity sensor values of the Thymio
                more info: http://wiki.thymio.org/en:thymioapi#toc2
            emergency_stream (multiprocessing.Queue): stream to receive real time emergenecy status and proximity
                sensor values
            turn_avoid_angle: angle to turn from obstacle to try to avoid it
        Returns:
            None
        Note: there is a recursive mutual call between this function and turn_robot. The calls are going to continue
            until the proximity sensor values clear up fully.
    """
    if turn_avoid_angle is None:
        turn_avoid_angle = control.OBSTACLE_TURN_ANGLE

    if isinstance(prox_vals, list):
        prox_vals = np.array(prox_vals)

    # any of the front sensors are on
    if np.any(prox_vals[0:5] > 0):

        # any of the back sensors are also on, send warning, we might be locked
        if np.any(prox_vals[5:7]>0):
            logger.warning(f'ROBOT seems to be locked, proximity values: {prox_vals}')
            pass

        # some of the front sensors are on
        # act according to the closest point/sensor with maximal value
        closest_sensor = np.argmax(prox_vals[0:5])
        logger.debug(f'Sensor with highest value: {closest_sensor}, with value {prox_vals[closest_sensor]}')

        # # left sensors on, avoid obstacle by turning right
        # if closest_sensor in [0, 1]:
        #     turn_robot(network, turn_avoid_angle, emergency_stream)
        #
        # # middle sensor is on, close to orthogonal collision is expected
        # elif closest_sensor == 2:

        # check which direction we deviate from orthogonal to turn properly
        left_proximity = np.mean(prox_vals[0:2])
        right_proximity = np.mean(prox_vals[3:5])

        # symmetric proximity, we have a wall in front that we can not pass but has a hole in the middle, or a corner
        if np.abs(left_proximity-right_proximity) < 500:
            # keep rotational direction and keep rotating
            logger.warning("SYMMETRIC OBSTACLES!!!")
            # current_rotation_direction = np.sign(network.GetVariable("thymio-II", "motor.left.speed")[0])
            # turn_robot(network, current_rotation_direction * turn_avoid_angle, emergency_stream)
            move_robot(network, 'Backward', 25, emergency_stream)
        if left_proximity > right_proximity:
            turn_robot(network, turn_avoid_angle, emergency_stream)
        else:
            turn_robot(network, -turn_avoid_angle, emergency_stream)

        # # right sensors on, avoid obstacle by turning left
        # elif closest_sensor in [3, 4]:
        #     turn_robot(network, -turn_avoid_angle, emergency_stream)

    # IGNORED FOR NOW AS ONLY BACK SENSORS NEVER TRIGGER EMERGENCY MODE
    # none of the front sensors are on
    else:
        # both back sensors are signaling (currently not implemented, we just exit avoidance protocol)
        if np.all(prox_vals[5:7]>0):
            return ("End avoidance")
            # return "Speed up", 1.5

        # only back left is signalling
        elif prox_vals[5]>0:
            turn_robot(network, turn_avoid_angle, emergency_stream)

        # only back right is signalling
        elif prox_vals[6]>0:
            turn_robot(network, -turn_avoid_angle, emergency_stream)


def run_additional_protocol(network, additional_protocol, emergency_stream):

    protocol_name = additional_protocol[0]
    if protocol_name == "Speed up":
        speed_up_robot(network, additional_protocol[1], emergency_stream)
    elif protocol_name == "End avoidance":
        return


def avoid_obstacle(network, prox_vals, emergency_stream):
    light_up_led(network, 32, 0, 0)
    # TODO: check proximity values and if obstacle is too close do backwards movement
    # TODO: keep velocity that the robot had when enetered in obstacle avoidance mode
    additional_protocol = turn_avoid_obstacle(network, prox_vals, emergency_stream)
    if additional_protocol is not None:
        logger.info('ADDITIONAL PROTOCOL AFTER TURN')
        run_additional_protocol(network, additional_protocol, emergency_stream)
    else:
        # always moving a bit forward to close avoidance to be sure that the way is clear
        logger.info('MOVE ROBOT FWD')
        move_robot(network, 'Forward', 20, emergency_stream)
    logger.info('Obstacle Avoidance Protocol done!')


def control_thymio(control_stream, motor_control_mode_stream, emergency_stream, with_control=False):
    """
    Process to switch between movement regimes and control the movement of Thymio2 robot via DBUS.
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
                try:
                    (emergency_mode, proximity_values) = emergency_stream.get_nowait()
                except Empty:
                    pass

                if not emergency_mode:
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

                else:
                    # triggering obstacle avoidance system
                    avoid_obstacle(network, proximity_values, emergency_stream)

                    # emptying accumulated queues
                    empty_queue(control_stream)
                    empty_queue(motor_control_mode_stream)
                    empty_queue(emergency_stream)

                    # turn off emergency mode and return to normal mode
                    emergency_mode = False
                    if movement_mode == "EXPLORE":
                        light_up_led(network, expR, expG, expB)
                    elif movement_mode == "BEHAVE":
                        light_up_led(network, behR, behG, behB)

                # To test infinite loops
                if env.EXIT_CONDITION:
                    break
        else:
            logger.error(f'{bcolors.FAIL}🗴 CONNECTION FAILED{bcolors.ENDC} via asebamedulla')
            motorinterface.asebamedulla_end()
            raise Exception('asebamedulla connection not healthy!')


def emergency_behavior(emergency_stream):
    """
    Process to check for emergency signals via proximity sensors and transmit information to other processes
        Args:
            emergency_stream (multiprocessing Queue): stream to push emergency status and sensor values
        Returns:
            -shall not return-
    """
    # Initializing DBus
    dbus.mainloop.glib.threads_init()
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    # Create Aseba network
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                             dbus_interface='ch.epfl.mobots.AsebaNetwork')

    t = datetime.now()
    while True:
        # enforcing checks on a regular basis
        if abs(t-datetime.now()).total_seconds() > (1 / control.EMERGENCY_CHECK_FREQ):

            # reading proximity values
            prox_val = np.array([val for val in network.GetVariable("thymio-II", "prox.horizontal")])

            if np.any(prox_val[0:5] > control.EMERGENCY_PROX_THRESHOLD):
                logger.info('Triggered Obstacle Avoidance!')
                emergency_stream.put((True, prox_val))
            else:
                emergency_stream.put((False, None))

            t =datetime.now()
