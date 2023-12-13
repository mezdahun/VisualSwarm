from visualswarm.contrib import simulation

if not simulation.ENABLE_SIMULATION:
    import dbus
    import dbus.mainloop.glib
    from visualswarm.control import motorinterface

import logging
from visualswarm.contrib import logparams, control, behavior, physconstraints, monitoring
from visualswarm.contrib import algorithm_improvements as algoimp
from visualswarm import env

import numpy as np
import tempfile
from datetime import datetime
from queue import Empty

# using main logger
if not simulation.ENABLE_SIMULATION:
    # setup logging
    import os
    ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
    logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
    logger.setLevel(monitoring.LOG_LEVEL)
else:
    logger = logging.getLogger('visualswarm.app_simulation')  # pragma: simulation no cover
bcolors = logparams.BColors


def rgb_to_hex(R, G, B):  # pragma: simulation no cover
    R = int((R / 32) * 200)
    G = int((G / 32) * 200)
    B = int((B / 32) * 200)
    return '0x%02x%02x%02x' % (R, G, B)


def light_up_led(network, R, G, B, webots_do_stream=None):
    """
    Method to indicate movement mode by lighting up top LEDS on robot
        Args:
            network: DBUS network to reach Thymio2
            R, G, B: color configuration of led, min: (0, 0, 0), max: (32, 32, 32)
        Returns:
            None
    """
    if not simulation.ENABLE_SIMULATION:
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
    else:
        webots_do_stream.put(('LIGHTUP_LED', int(rgb_to_hex(R, G, B), 0)))  # pragma: simulation no cover


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


def rotate(rot_to_right=None, v_rot=None) -> list:
    """
    Method to get motor velocity values according to a preconfigured rotation (ROT) process
        Args:
            No args, configured via contrib.control
        Returns:
            [v_left, v_right]: ROT motor values
    """
    if rot_to_right is None:
        if control.ROT_DIRECTION == 'Left':
            right_sign = 1
        elif control.ROT_DIRECTION == 'Right':
            right_sign = -1
        elif control.ROT_DIRECTION == 'Random':
            right_sign = np.random.choice([1, -1], 1)[0]
        else:
            logger.error(f"Wrong configuration value control.ROT_DIRECTION=\"{control.ROT_DIRECTION}\"! Abort!")
            raise KeyboardInterrupt
    else:
        if rot_to_right:
            right_sign = -1
        else:
            right_sign = 1

    left_sign = -1 * right_sign

    if not algoimp.WITH_EXPLORE_ROT:
        v_left = left_sign * control.ROT_MOTOR_SPEED
        v_right = right_sign * control.ROT_MOTOR_SPEED
    else:
        if v_rot is None:
            v_left = left_sign * algoimp.EXPLORE_ROT_SPEED
            v_right = right_sign * algoimp.EXPLORE_ROT_SPEED
        else:
            v_left = left_sign * v_rot
            v_right = right_sign * v_rot

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


def limit_front_back_movement(v):
    """Limiting backwards movement speed if requested to avoid collisions."""
    if v < 0:
        if algoimp.WITH_LIMITED_BACKWARDS and v < -algoimp.MAX_BACKWARDS_SPEED:
            print(f"Limiting velocity {v} to {algoimp.MAX_BACKWARDS_SPEED}")
            v = -algoimp.MAX_BACKWARDS_SPEED
    else:
        if algoimp.WITH_LIMITED_FORWARD and v > algoimp.MAX_FORWARD_SPEED:
            print(f"Limiting velocity {v} to {algoimp.MAX_FORWARD_SPEED}")
            v = algoimp.MAX_FORWARD_SPEED
    return v


def distribute_overall_speed(v: float, dpsi: float, excl=None, excr=None, v_thr=20) -> list:
    """
    distributing desired forward speed to motor velocities according to the change in the heading angle dpsi.
        Args:
            v (float): desired forward speed of the agent
            dpsi (float): change in the heading angle of the agent
        Returns:
            [v_left, v_right]: motor velocity values of the agent
    """

    # # Calculating proportional heading angle change
    dpsi_p = dpsi / np.pi

    # trying to distinguish attraction vs repulsion
    # by checking if the agent is turning towards or away from the cue
    is_right = excr > excl

    # Stationary turning to avoid lazy turning with lower speeds
    # and to avoid loosing other agents from sight due to repulsion forces
    if algoimp.WITH_STAT_TURNING:
        if v < 0 and algoimp.WITH_LIMITED_BACKWARDS and v < algoimp.MAX_BACKWARDS_SPEED:
            # stationary turn due to large angle and low speed
            print(f"velocity: {v}")
            print(f"prop. angle change: {dpsi_p}")
            # dpsi_p = np.sign(dpsi_p) * min(np.abs(dpsi_p), 0.001)
            if excl is not None:
                # turning towards more retinal excitation
                dpsi_p = ((excr - excl) / (excr + excl)) * 0.001
            v_left = - algoimp.MAX_BACKWARDS_SPEED + algoimp.STAT_TURN_SPEED_BACK * dpsi_p
            v_right = - algoimp.MAX_BACKWARDS_SPEED - algoimp.STAT_TURN_SPEED_BACK * dpsi_p
            # if excl is not None:
            #     if (is_right and v_left > v_right) or (not is_right and v_right >= v_left):
            #         # attraction we keep moving as before
            #         pass
            #     else:
            #         # reulsion we reverse the turning, soven if something is very close we keep turning towards it
            #         v_left, v_right = v_right, v_left

        elif np.abs(v) < algoimp.STAT_TURN_VEL_THRES:
            # stationary turn due to large angle and low speed
            print(f"velocity: {v}")
            print(f"prop. angle change: {dpsi_p}")
            # dpsi_p = np.sign(dpsi_p) * min(np.abs(dpsi_p), 0.025)
            print(f"prop. angle change: {dpsi_p}")
            if excl is not None:
                # turning towards more retinal excitation
                dpsi_p = ((excr - excl) / (excr + excl)) * 0.001
            v_left = + algoimp.STAT_TURN_SPEED * dpsi_p
            v_right = - algoimp.STAT_TURN_SPEED * dpsi_p

        else:
            # Limiting backwards movement speed if requested
            if algoimp.WITH_LIMITED_BACKWARDS:
                v = limit_front_back_movement(v)

            # Matching simulation scale with reality
            v = v * behavior.KAP

            # Distributing velocity
            v_left = v * (1 + dpsi_p)
            v_right = v * (1 - dpsi_p)
    else:
        # Limiting backwards movement speed if requested
        if algoimp.WITH_LIMITED_BACKWARDS:
            v = limit_front_back_movement(v)

        # Matching simulation scale with reality
        v = v * behavior.KAP

        # Distributing velocity
        v_left = v * (1 + dpsi_p)
        v_right = v * (1 - dpsi_p)

    return [v_left, v_right]


def get_latest_element(queue):  # pragma: simulation no cover
    """
    fetching the latest element in queue and by that emptying the FIFO Queue object. Use this to consume queue elements
    with a slow process that is filled up by a faster process
        Args:
            queue (multiprocessing.Queue): queue object to be emptied and returned the latest element
        Returns:
            val: latest vaue in the queue
    """
    val = None
    while not queue.empty():
        try:
            val = queue.get_nowait()
        except Empty:
            return val
    return val


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
        except Empty:  # pragma: no cover
            return True
    return True


def turn_robot(network, angle, emergency_stream, turning_motor_speed=125, blind_mode=False,
               webots_do_stream=None):
    """
    turning robot with a specified speed to a particular physical angle according to the heuristics (multipliers)
    defined in contrib.physconstraints
        Args:
            network (dbus network): network on which we communicate with Thymio
            angle (float or int): physical angle in degrees to turn the robot with
            emergency_stream (multiprocessing.Queue): stream to receive real time emergenecy status and proximity
                sensor values
            turning_motor_speed (int), optional: motor speed to turn the robot with
            blind_mode (bool), optional: if blind is true, the recursive function call will not be activated
        Returns:
            None
        Note: recursively calling turn_avoid_obstacle if obstacle is detected during turning as well as this
            method is called from turn_avoid_obstacle. As a result the recursion is continued until the proximity
            sensors are free.
    """
    # TODO: implement turning with keeping forward speed that can be calculated from v_left and v_right as avg
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
            if not simulation.ENABLE_SIMULATION:
                network.SetVariable("thymio-II", "motor.left.target", [np.sign(angle) * turning_motor_speed])
                network.SetVariable("thymio-II", "motor.right.target", [-np.sign(angle) * turning_motor_speed])
            else:  # pragma: simulation no cover
                webots_do_stream.put(("SET_MOTOR", {'left': float(np.sign(angle) * turning_motor_speed),
                                                    'right': float(-np.sign(angle) * turning_motor_speed)}))

        else:
            # if the defined angle of turn was not enough to clear the proximity sensors we retry to recursively
            # call the turning maneuver according to the new proximity values. The frequency of this check is
            # restricted by the frequency of new elements in the emergency_stream, as the get method will wait for the
            # new element
            if not blind_mode:
                logger.debug('Recursive turning maneuver during obstacle detection...')
                turn_avoid_obstacle(network, proximity_values, emergency_stream, webots_do_stream=webots_do_stream)
                break
            else:
                logger.warning(f'Blind mode activated during turning {angle} degrees')
                logger.warning('Further emergency signals ignored!')

        # update emergency status and proximity values from emergency stream with wait behavior (get).
        (recursive_obstacle, proximity_values) = emergency_stream.get()


def move_robot(network, direction, distance, emergency_stream, moving_motor_speed=125, blind_mode=False,
               webots_do_stream=None):
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
            blind_mode (bool), optional: if blind is true, the recursive function call will not be activated
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
        logger.error(f'Unknown movement direction: {direction}')
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
            if not simulation.ENABLE_SIMULATION:
                network.SetVariable("thymio-II", "motor.left.target", [movesign * moving_motor_speed])
                network.SetVariable("thymio-II", "motor.right.target", [movesign * moving_motor_speed])
            else:  # pragma: simulation no cover
                webots_do_stream.put(("SET_MOTOR", {'left': float(movesign * moving_motor_speed),
                                                    'right': float(movesign * moving_motor_speed)}))

        else:
            if not blind_mode:
                avoid_obstacle(network, proximity_values, emergency_stream, webots_do_stream=webots_do_stream)
                break
            else:
                logger.warning(f'Blind mode activated during moving {direction}, emergency signal ignored!')

        (recursive_obstacle, proximity_values) = emergency_stream.get()


def stop_robot(network, duration, webots_do_stream=None):
    """Settin robot speed to zero."""
    start_time = datetime.now()
    while abs(start_time - datetime.now()).total_seconds() < duration:
        if not simulation.ENABLE_SIMULATION:
            network.SetVariable("thymio-II", "motor.left.target", [0])
            network.SetVariable("thymio-II", "motor.right.target", [0])
        else:  # pragma: simulation no cover
            webots_do_stream.put(("SET_MOTOR", {'left': float(0),
                                                'right': float(0)}))

def speed_up_robot(network, additional_motor_speed_multiplier,  # pragma: no cover
                   emergency_stream, protocol_time=0.5):
    """
    speeding up robot with a specified motor speed multiplier for a given time.
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
            until the proximity sensors are free, after which all recursively called avoid_obstacle methods will
            return.
    """
    # # first getting current motor values
    # v_left_curr = network.GetVariable("thymio-II", "motor.left.speed")[0]
    # v_right_curr = network.GetVariable("thymio-II", "motor.right.speed")[0]
    # logger.info(v_left_curr)
    # logger.info(v_right_curr)
    #
    # v_left_target = additional_motor_speed_multiplier * v_left_curr
    # v_right_target = additional_motor_speed_multiplier * v_right_curr
    #
    # # from this point the method shows a lot of similarity with turn_robot, please check the comments there
    # empty_queue(emergency_stream)
    #
    # recursive_obstacle = False
    # proximity_values = None
    #
    # start_time = datetime.now()
    # while abs(start_time - datetime.now()).total_seconds() < protocol_time:
    #
    #     if not recursive_obstacle:
    #         network.SetVariable("thymio-II", "motor.left.target", [v_left_target])
    #         network.SetVariable("thymio-II", "motor.right.target", [v_right_target])
    #
    #     else:
    #         # TODO: check if we are locked and return if yes
    #         avoid_obstacle(network, proximity_values, emergency_stream)
    #
    #     (recursive_obstacle, proximity_values) = emergency_stream.get()
    pass  # pragma: no cover


def turn_avoid_obstacle(network, prox_vals, emergency_stream, turn_avoid_angle=None,
                        webots_do_stream=None):
    """
    deciding on direction and starting turning maneuver during obstacle avoidance.
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

    # transforming prox_vals to array if not in desired format
    if isinstance(prox_vals, list):
        prox_vals = np.array(prox_vals)

    # any of the back sensors are on
    if np.any(prox_vals[5::] > control.EMERGENCY_PROX_THRESHOLD_BACK):
        # Stop the robot
        return "Stop", 0.5

    # any of the front sensors are on
    if np.any(prox_vals[0:5] > 0):

        # any of the back sensors are also on, send warning as we might be locked
        # but proceed according to frontal sensors
        if np.any(prox_vals[5:7] > 0):
            logger.warning(f'Agent might be locked! Proximity values: {prox_vals}')

        if control.AVOID_TURN_DIRECTION == 'Various':
            # check which direction we deviate from orthogonal to decide on turning direction
            left_proximity = np.mean(prox_vals[0:2])
            right_proximity = np.mean(prox_vals[3:5])
            logger.debug(f'Frontal center prox: {prox_vals[2]}')
            logger.debug(f'Left sum prox: {left_proximity} vs Right sum prox: {right_proximity}')

            # Pendulum Trap (corner or symmetric non-continuous obstacle around the robot)
            if np.abs(left_proximity - right_proximity) < control.SYMMETRICITY_THRESHOLD and \
                    prox_vals[2] < control.UNCONTINOUTY_THRESHOLD:
                logger.warning("Pendulum trap strategy initiated!")
                # change orientation (always to the right) drastically to get out of pendulum trap
                turn_robot(network, control.PENDULUM_TRAP_ANGLE, emergency_stream, blind_mode=True,
                           webots_do_stream=webots_do_stream)
                return "Move", "Forward", 20

            # Obstacle is closer to the left, turn right
            elif left_proximity > right_proximity:
                turn_robot(network, turn_avoid_angle, emergency_stream, webots_do_stream=webots_do_stream)
                return "Move", "Forward", 20

            # Obstacle is closer to the right, turn left
            else:
                turn_robot(network, -turn_avoid_angle, emergency_stream, webots_do_stream=webots_do_stream)
                return "Move", "Forward", 20

        elif control.AVOID_TURN_DIRECTION == 'Uniform':
            # In case we always want the agent to turn the same direction during avoidance
            # even if this yields a longer avoidance time
            turn_robot(network, turn_avoid_angle, emergency_stream, webots_do_stream=webots_do_stream)
            return "Move", "Forward", 40

    # IGNORED FOR NOW AS ONLY BACK SENSORS NEVER TRIGGER EMERGENCY MODE
    # none of the front sensors are on
    # else:
    #     # both back sensors are signaling (currently not implemented, we just exit avoidance protocol)
    #     if np.all(prox_vals[5:7]>0):
    #         return ("End avoidance")
    #         # return "Speed up", 1.5
    #
    #     # only back left is signalling
    #     elif prox_vals[5]>0:
    #         turn_robot(network, turn_avoid_angle, emergency_stream)
    #
    #     # only back right is signalling
    #     elif prox_vals[6]>0:
    #         turn_robot(network, -turn_avoid_angle, emergency_stream)


def run_additional_protocol(network, additional_protocol, emergency_stream,
                            webots_do_stream=None):
    """
    Running additional necessary protocol after turning the robot.
        Args:
            network (dbus network): network on which we communicate with Thymio
            additional_protocol (list): first element is the name of the protocol, additional elements are the arguments
                of the initiated additional protocol.
            emergency_stream (multiprocessing.Queue): stream to receive real time emergenecy status and proximity
                sensor values
        Returns:
            None
    """
    protocol_name = additional_protocol[0]
    if protocol_name == "Speed up":
        speed_up_robot(network, additional_protocol[1], emergency_stream)
    elif protocol_name == "Move":
        move_robot(network, additional_protocol[1], additional_protocol[2], emergency_stream,
                   webots_do_stream=webots_do_stream)
    elif protocol_name == "Stop":
        stop_robot(network, additional_protocol[1], webots_do_stream=webots_do_stream)
    elif protocol_name == "End avoidance":
        return


def avoid_obstacle(network, prox_vals, emergency_stream,
                   webots_do_stream=None):
    """
    Initiating 2-level recursive obstacle avoidance algorithm
        Args:
            network (dbus network): network on which we communicate with Thymio
            prox_vals (list or np.array): len 7 array with the proximity sensor values of the Thymio
                more info: http://wiki.thymio.org/en:thymioapi#toc2
            emergency_stream (multiprocessing.Queue): stream to receive real time emergenecy status and proximity
                sensor values
        Returns:
            None
    """
    # TODO: keep velocity that the robot had when entered in obstacle avoidance mode
    additional_protocol = turn_avoid_obstacle(network, prox_vals, emergency_stream, webots_do_stream=webots_do_stream)

    if additional_protocol is not None:
        logger.info(f'Initiated additional protocol after turn: {additional_protocol}')
        run_additional_protocol(network, additional_protocol, emergency_stream, webots_do_stream=webots_do_stream)

    logger.info('Obstacle Avoidance Protocol done!')


def control_thymio(control_stream, motor_control_mode_stream, emergency_stream, with_control=False,
                   webots_do_stream=None):
    """
    Process to switch between movement regimes and control the movement of Thymio2 robot via DBUS.
        Args:
            control_stream (multiprocessing Queue): stream to push calculated control parameters
            motor_control_mode_stream (multiprocessing Queue): stream to get movement type/mode.
            with_control (boolean): sends motor command to robot if true. Only consumes input stream if false.
            webots_do_stream (multiprocessing Queue): if we use a webots simulation this stream is to communicate with
                webots interface and set motor values with pushed values.
        Returns:
            -shall not return-
    """
    try:
        prev_movement_mode = "BEHAVE"
        (emergR, emergG, emergB) = control.EMERGENCY_STATUS_RGB
        (expR, expG, expB) = control.EXPLORE_STATUS_RGB
        (behR, behG, behB) = control.BEHAVE_STATUS_RGB

        if not with_control:
            # simply consuming the input stream so that we don't fill up memory
            while True:
                (v, dpsi, el, er) = control_stream.get()
                movement_mode = motor_control_mode_stream.get()

                # To test infinite loops
                if env.EXIT_CONDITION:
                    break
        else:
            # Initialize timestamp
            last_explore_change = datetime.now()
            last_behave_change = datetime.now()

            if not simulation.ENABLE_SIMULATION:
                # Initializing DBus
                dbus.mainloop.glib.threads_init()
                dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
                bus = dbus.SessionBus()

                # Create Aseba network
                network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                         dbus_interface='ch.epfl.mobots.AsebaNetwork')

                is_connection_healthy = motorinterface.asebamedulla_health(network)
            else:  # pragma: simulation no cover
                network = "SimulationDummy"
                is_connection_healthy = True

            if is_connection_healthy:
                logger.info(f'{bcolors.OKGREEN}✓ CONNECTION SUCCESSFUl{bcolors.ENDC} via asebamedulla')

                v_last = 0
                dpsi_last = 0
                while True:
                    # fetching state variables
                    (v, dpsi, excl, excr) = control_stream.get()
                    movement_mode = motor_control_mode_stream.get()
                    try:
                        latest_emergency = get_latest_element(emergency_stream)
                        if latest_emergency is not None:
                            (emergency_mode, proximity_values) = latest_emergency
                        else:
                            emergency_mode = False
                    except Empty:  # pragma: no cover
                        emergency_mode = False

                    if not emergency_mode:
                        if movement_mode == "BEHAVE":

                            # Switch between modes, change mode status LED (can not be tested with single cycle)
                            if prev_movement_mode == "EXPLORE":  # pragma: no cover
                                light_up_led(network, behR, behG, behB, webots_do_stream=webots_do_stream)

                            # Persistent change in movement mode
                            is_persistent = abs((last_explore_change - datetime.now()).total_seconds()) > \
                                            control.WAIT_BEFORE_SWITCH_MOVEMENT
                            if is_persistent or env.EXIT_CONDITION:
                                v_last = v
                                dpsi_last = dpsi
                                # Behavior according to Romanczuk and Bastien 2020
                                # distributing desired forward speed according to dpsi
                                [v_left, v_right] = distribute_overall_speed(v, dpsi, excl=excl, excr=excr)

                                # hard limit motor velocities but keep their ratio for desired movement
                                if np.abs(v_left) > control.MAX_MOTOR_SPEED or \
                                        np.abs(v_right) > control.MAX_MOTOR_SPEED:
                                    logger.warning(f'Reached max velocity: left:{v_left:.2f} right:{v_right:.2f}')
                                    #[v_left, v_right] = hardlimit_motor_speed(v_left, v_right)

                                # sending motor values to robot
                                if not simulation.ENABLE_SIMULATION:
                                    if behavior.MOVE_IN_CIRCLE:
                                        v_left = 125
                                        v_right = 110
                                    network.SetVariable("thymio-II", "motor.left.target", [v_left])
                                    network.SetVariable("thymio-II", "motor.right.target", [v_right])
                                else:   # pragma: simulation no cover
                                    webots_do_stream.put(("SET_MOTOR", {'left': v_left, 'right': v_right}))

                                logger.debug(f"BEHAVE left: {v_left} \t right: {v_right}")
                                # last time we changed velocity according to BEHAVIOR REGIME
                                last_behave_change = datetime.now()
                            else:
                                [v_left, v_right] = distribute_overall_speed(v_last, dpsi_last, excl=excl, excr=excr)
                                if not simulation.ENABLE_SIMULATION:
                                    if behavior.MOVE_IN_CIRCLE:
                                        v_left = 125
                                        v_right = 110
                                    network.SetVariable("thymio-II", "motor.left.target", [v_left])
                                    network.SetVariable("thymio-II", "motor.right.target", [v_right])
                                else:   # pragma: simulation no cover
                                    webots_do_stream.put(("SET_MOTOR", {'left': v_left, 'right': v_right}))


                        elif movement_mode == "EXPLORE":

                            # Switch between modes, change mode status LED
                            if prev_movement_mode == "BEHAVE":
                                light_up_led(network, expR, expG, expB, webots_do_stream=webots_do_stream)

                            # Persistent change in modes
                            if abs((last_behave_change - datetime.now()).total_seconds()) \
                                    > control.WAIT_BEFORE_SWITCH_MOVEMENT:
                                # Enforcing specific dt in Random Walk Process
                                if abs((last_explore_change - datetime.now()).total_seconds()) > 0:  #control.RW_DT:

                                    # Dumm exploration techniques
                                    if not algoimp.WITH_EXPLORE_ROT and not algoimp.WITH_EXPLORE_ROT_CONT:
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

                                    # Improved exploration rotation towards the last scial cue
                                    # checking the direction of last social cue with checking dpsi sign
                                    else:
                                        if algoimp.WITH_EXPLORE_ROT:
                                            # Using stationary rotation
                                            logger.debug(f'dorientation: {dpsi_last}')
                                            rot_to_right = np.sign(dpsi_last) <= 0
                                            logger.debug(f'rot_to_right: {rot_to_right}')
                                            [v_left, v_right] = rotate(rot_to_right=rot_to_right)
                                            logger.debug("Improved exploration rotation towards the last social cue")
                                        elif algoimp.WITH_EXPLORE_ROT_CONT:
                                            # # Using continous movement
                                            # The rotation should happen in the direction of last calculated direction
                                            # but is a fixed rate always
                                            dpsi_rot = np.sign(dpsi_last) * algoimp.EXPLORE_ROT_THETA_CONT
                                            # If no direction is defined we turn right
                                            if dpsi_rot == 0:
                                                dpsi_rot = algoimp.EXPLORE_ROT_THETA_CONT
                                            # rotation happens with the last speed except if it's zero
                                            v_rot = algoimp.EXPLORE_ROT_SPEED_CONT if v_last == 0 else v_last
                                            logger.info(f"dpsi_last: {dpsi_last} \t dpis_rot: {dpsi_rot} \t v_rot: {v_rot}")
                                            [v_left, v_right] = distribute_overall_speed(v_rot, dpsi_rot)


                                    logger.debug(f'EXPLORE left: {v_left} \t right: {v_right}')

                                    # sending motor values to robot
                                    if not simulation.ENABLE_SIMULATION:
                                        network.SetVariable("thymio-II", "motor.left.target", [v_left])
                                        network.SetVariable("thymio-II", "motor.right.target", [v_right])
                                    else:   # pragma: simulation no cover
                                        webots_do_stream.put(("SET_MOTOR",
                                                              {'left': float(v_left), 'right': float(v_right)}))

                                    # last time we changed velocity according to EXPLORE REGIME
                                    last_explore_change = datetime.now()
                            else:
                                # if the movement is not yet persistent we continue to move according to BEHAVE
                                # regime
                                [v_left, v_right] = distribute_overall_speed(v_last, dpsi_last, excl=excl, excr=excr)

                                # sending motor values to robot
                                if not simulation.ENABLE_SIMULATION:
                                    network.SetVariable("thymio-II", "motor.left.target", [v_left])
                                    network.SetVariable("thymio-II", "motor.right.target", [v_right])
                                else:  # pragma: simulation no cover
                                    webots_do_stream.put(("SET_MOTOR",
                                                          {'left': float(v_left), 'right': float(v_right)}))

                        else:
                            logger.error(f"Unknown movement type \"{movement_mode}\"! Abort!")
                            raise KeyboardInterrupt

                        prev_movement_mode = movement_mode

                    else:
                        # showing emergency mode with top LEDs
                        light_up_led(network, emergR, emergG, emergB, webots_do_stream=webots_do_stream)
                        # triggering obstacle avoidance system
                        avoid_obstacle(network, proximity_values, emergency_stream, webots_do_stream=webots_do_stream)

                        # emptying accumulated queues
                        empty_queue(control_stream)
                        empty_queue(motor_control_mode_stream)
                        empty_queue(emergency_stream)

                        # turn off emergency mode and return to normal mode, showing this with LEDs
                        emergency_mode = False
                        if movement_mode == "EXPLORE":  # pragma: no cover
                            light_up_led(network, expR, expG, expB, webots_do_stream=webots_do_stream)
                        elif movement_mode == "BEHAVE":
                            light_up_led(network, behR, behG, behB, webots_do_stream=webots_do_stream)

                    # To test infinite loops
                    if env.EXIT_CONDITION:
                        break
            else:
                if not simulation.ENABLE_SIMULATION:
                    logger.error(f'{bcolors.FAIL}🗴 CONNECTION FAILED{bcolors.ENDC} via asebamedulla')
                    motorinterface.asebamedulla_end()
                    raise Exception('asebamedulla connection not healthy!')
    except KeyboardInterrupt:
        pass


def emergency_behavior(emergency_stream, sensor_stream=None):
    """
    Process to check for emergency signals via proximity sensors and transmit information to other processes
        Args:
            emergency_stream (multiprocessing Queue): stream to push emergency status and sensor values
            sensor_stream (multiprocessing Queue): in case of webots simulation this stream should be continously
                updated with the virtual robots sensor values
        Returns:
            -shall not return-
    """
    try:
        if not simulation.ENABLE_SIMULATION:
            # Initializing DBus
            dbus.mainloop.glib.threads_init()
            dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
            bus = dbus.SessionBus()

            # Create Aseba network
            network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                     dbus_interface='ch.epfl.mobots.AsebaNetwork')

        t = datetime.now()
        if simulation.ENABLE_SIMULATION and sensor_stream is not None:   # pragma: simulation no cover
            logger.info(f'START: len{sensor_stream.qsize()}')
            empty_queue(sensor_stream)

        # saving previous proximity values to only react to consistent proximity signals
        # that last at least 3 time steps
        prev_prev_prox = np.zeros(7)
        prev_prox = np.zeros(7)
        while True:
            # enforcing checks on a regular basis
            if abs(t - datetime.now()).total_seconds() > (1 / control.EMERGENCY_CHECK_FREQ):

                # reading proximity values
                if not simulation.ENABLE_SIMULATION:
                    prox_val = np.array([val for val in network.GetVariable("thymio-II", "prox.horizontal")])
                else:   # pragma: simulation no cover
                    if sensor_stream is not None:
                        prox_val = np.array(get_latest_element(sensor_stream))
                    else:
                        raise Exception('No sensor stream has been passed from Webots to sentinel process!')

                try:
                    # TODO: think over filtering of IR signals
                    if np.any(prox_val[0:5] > control.EMERGENCY_PROX_THRESHOLD) \
                            and np.any(prev_prox[0:5] > control.EMERGENCY_PROX_THRESHOLD) \
                            and np.any(prev_prev_prox[0:5] > control.EMERGENCY_PROX_THRESHOLD):
                        logger.info('Triggered Obstacle Avoidance!')
                        emergency_stream.put((True, prox_val))
                    elif np.any(prox_val[5::] > control.EMERGENCY_PROX_THRESHOLD_BACK) \
                            and np.any(prev_prox[5::] > control.EMERGENCY_PROX_THRESHOLD_BACK) \
                            and np.any(prev_prev_prox[5::] > control.EMERGENCY_PROX_THRESHOLD_BACK):
                        logger.info('Triggered obstacle avoidance from back!')
                        emergency_stream.put((True, prox_val))
                    else:
                        emergency_stream.put((False, None))
                    prev_prev_prox = prev_prox
                    prev_prox = prox_val
                except IndexError:   # pragma: no cover
                    logger.warning('IndexError in sentinel process!!!')

                t = datetime.now()

            # To test infinite loops
            if env.EXIT_CONDITION:
                break

    except KeyboardInterrupt:
        pass
