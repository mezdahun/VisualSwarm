from unittest import TestCase, mock

from freezegun import freeze_time

from visualswarm.control import motoroutput


class MotorInterfaceTest(TestCase):

    @freeze_time("Jan 15th, 2020", auto_tick_seconds=15)
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    @mock.patch('dbus.mainloop.glib.DBusGMainLoop', return_value=None)
    @mock.patch('dbus.SessionBus')
    @mock.patch('dbus.Interface')
    @mock.patch('visualswarm.control.motoroutput.distribute_overall_speed', return_value=(10, 10))
    @mock.patch('visualswarm.control.motoroutput.hardlimit_motor_speed', return_value=(1, 1))
    @mock.patch('visualswarm.control.motoroutput.avoid_obstacle', return_value=None)
    @mock.patch('visualswarm.control.motoroutput.empty_queue', return_value=None)
    def test_control_thymio(self, mock_empty_queue, mock_avoid_obstacle, mock_hard_limit, mock_distribute,
                            mock_network_init, mock_dbus_sessionbus, mock_dbus_init):
        mock_network = mock.MagicMock(return_value=None)
        mock_network.SetVariable.return_value = None
        mock_network_init.return_value = mock_network

        mock_dbus_sessionbus.return_value.get_object.return_value = None

        # mocking streams
        control_stream = mock.MagicMock()
        control_stream.get.return_value = (1, 1)
        movement_mode_stream = mock.MagicMock()
        movement_mode_stream.get.return_value = "BEHAVE"
        emergency_stream = mock.MagicMock()
        emergency_stream.get_nowait.return_value = (False, None)

        # Case 1 : healthy connection via asebamedulla
        with mock.patch('visualswarm.control.motorinterface.asebamedulla_health') as mock_health:
            mock_health.return_value = True
            # Case 1/a : with no control
            motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream, with_control=False)
            control_stream.get.assert_called_once()
            movement_mode_stream.get.assert_called_once()

            control_stream.get.reset_mock()
            movement_mode_stream.get.reset_mock()
            mock_health.reset_mock()

            # Case 1/b : with control and normal output velcoity values
            with mock.patch('visualswarm.contrib.control.MAX_MOTOR_SPEED', 12):
                motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream, with_control=True)
                mock_dbus_init.assert_called_once_with(set_as_default=True)
                mock_dbus_sessionbus.assert_called_once()
                mock_network_init.assert_called_once()
                mock_health.assert_called_once()
                control_stream.get.assert_called_once()
                movement_mode_stream.get.assert_called_once()
                mock_distribute.assert_called_once()
                self.assertEqual(mock_network.SetVariable.call_count, 2)

            # Case 1/c : with control and too large output velcoity values
            with mock.patch('visualswarm.contrib.control.MAX_MOTOR_SPEED', 5):
                motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream, with_control=True)
                mock_hard_limit.assert_called_once()

            # Case 1/d: with control but emergency
            emergency_prox_values = [0, 25, 0, 0, 0, 0, 0]
            emergency_stream.get_nowait.return_value = (True, emergency_prox_values)
            with mock.patch('visualswarm.contrib.control.EMERGENCY_PROX_THRESHOLD', 20):
                with mock.patch('visualswarm.control.motoroutput.light_up_led') as mock_light:
                    motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream,
                                               with_control=True)
                    self.assertEqual(mock_light.call_count, 2)
                    mock_avoid_obstacle.assert_called_once()
                    self.assertEqual(mock_empty_queue.call_count, 3)

            # Case 1/e: with control but exploration
            movement_mode_stream.get.return_value = "EXPLORE"
            emergency_stream.get_nowait.return_value = (False, None)
            with mock.patch('visualswarm.contrib.control.WAIT_BEFORE_SWITCH_MOVEMENT', 15 - 1):
                # ROTATION
                with mock.patch('visualswarm.contrib.control.EXP_MOVE_TYPE', 'Rotation'):
                    with mock.patch('visualswarm.control.motoroutput.rotate') as mock_rotate:
                        mock_rotate.return_value = [0, 0]
                        motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream,
                                                   with_control=True)
                        mock_rotate.assert_called_once()

                # RANDOM WALK
                with mock.patch('visualswarm.contrib.control.EXP_MOVE_TYPE', 'RandomWalk'):
                    with mock.patch('visualswarm.control.motoroutput.step_random_walk') as mock_rw:
                        mock_rw.return_value = [0, 0]
                        motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream,
                                                   with_control=True)
                        mock_rw.assert_called_once()

                # UNKNOWN
                with mock.patch('visualswarm.contrib.control.EXP_MOVE_TYPE', 'Invalid'):
                    motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream,
                                               with_control=True)

            # Case 1/c: unknown movement mode
            movement_mode_stream = mock.MagicMock()
            movement_mode_stream.get.return_value = "UNKNOWN"
            motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream,
                                       with_control=True)

        # Case 2 : unhealthy connection via asebamedulla
        with mock.patch('visualswarm.control.motorinterface.asebamedulla_health') as mock_health:
            with mock.patch('visualswarm.control.motorinterface.asebamedulla_end'):
                mock_health.return_value = False
                control_stream.reset_mock()
                movement_mode_stream.get.reset_mock()
                motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream,
                                           with_control=False)
                control_stream.get.assert_called_once()
                movement_mode_stream.get.assert_called_once()

                with self.assertRaises(Exception):
                    motoroutput.control_thymio(control_stream, movement_mode_stream, emergency_stream,
                                               with_control=True)

    @freeze_time("Jan 15th, 2020", auto_tick_seconds=15)
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    @mock.patch('dbus.mainloop.glib.DBusGMainLoop', return_value=None)
    @mock.patch('dbus.SessionBus')
    @mock.patch('dbus.Interface')
    def test_emergency_behavior(self, mock_network_init, mock_dbus_sessionbus, mock_dbus_init):
        with mock.patch('visualswarm.contrib.control.EMERGENCY_CHECK_FREQ', 1):
            mock_network = mock.MagicMock(return_value=None)
            mock_network_init.return_value = mock_network

            mock_dbus_sessionbus.return_value.get_object.return_value = None

            # mocking streams
            emergency_stream = mock.MagicMock()
            emergency_stream.put.return_value = None

            # CASE1: no emergency
            mock_network.GetVariable.return_value = [0, 0, 0, 0, 0, 0, 0]
            motoroutput.emergency_behavior(emergency_stream)
            mock_network.GetVariable.assert_called_once_with("thymio-II", "prox.horizontal")
            emergency_stream.put.assert_called_once_with((False, None))

            # Case2: emergency on frontal sensors
            emergency_prox_values = [0, 25, 0, 0, 0, 0, 0]
            mock_network.reset_mock()
            mock_network.GetVariable.return_value = emergency_prox_values
            emergency_stream.put.reset_mock()
            with mock.patch('visualswarm.contrib.control.EMERGENCY_PROX_THRESHOLD', 20):
                motoroutput.emergency_behavior(emergency_stream)
                mock_network.GetVariable.assert_called_once_with("thymio-II", "prox.horizontal")
                emergency_stream.put.assert_called_once()
                args, kwargs = emergency_stream.put.call_args
                self.assertTrue(args[0][0])
                self.assertListEqual(list(args[0][1]), emergency_prox_values)

    @mock.patch('visualswarm.control.motoroutput.turn_avoid_obstacle')
    @mock.patch('visualswarm.control.motoroutput.run_additional_protocol', return_value=None)
    def test_avoid_obstacle(self, mock_run_add_prot, mock_turn_avoid):
        network = "mock network"
        prox_vals = [0, 0, 0, 0, 0, 0, 0]
        emergency_stream = "mock emergency stream"

        # case 1: turn returns none, no additional protocol
        mock_turn_avoid.return_value = None
        motoroutput.avoid_obstacle(network, prox_vals, emergency_stream)
        mock_turn_avoid.assert_called_once_with(network, prox_vals, emergency_stream)
        mock_run_add_prot.assert_not_called()

        # case 2: we get back an additional protocol
        mock_turn_avoid.reset_mock()
        mock_run_add_prot.reset_mock()
        mock_turn_avoid.return_value = "Some additional protocol"
        motoroutput.avoid_obstacle(network, prox_vals, emergency_stream)
        mock_turn_avoid.assert_called_once_with(network, prox_vals, emergency_stream)
        mock_run_add_prot.assert_called_once_with(network, 'Some additional protocol', emergency_stream)

    def test_rotate(self):
        with mock.patch('visualswarm.contrib.control.ROT_MOTOR_SPEED', 100):
            with mock.patch('visualswarm.contrib.control.ROT_DIRECTION', 'Left'):
                self.assertEqual(motoroutput.rotate(), [-100, 100])
            with mock.patch('visualswarm.contrib.control.ROT_DIRECTION', 'Right'):
                self.assertEqual(motoroutput.rotate(), [100, -100])
            with mock.patch('visualswarm.contrib.control.ROT_DIRECTION', 'Random'):
                with mock.patch('numpy.random.choice') as mock_choice:
                    mock_choice.return_value = [1]  # positive right motor (left rotation)
                    [v_l, v_r] = motoroutput.rotate()
                    mock_choice.assert_called_once()
                    self.assertEqual([v_l, v_r], [-100, 100])
            with self.assertRaises(KeyboardInterrupt):
                with mock.patch('visualswarm.contrib.control.ROT_DIRECTION', 'UNKNOWN'):
                    [v_l, v_r] = motoroutput.rotate()

    def test_light_up_led(self):
        tempfile_mock = mock.MagicMock()
        tempfile_mock.__enter__.return_value = mock.MagicMock()
        tempfile_mock.__enter__.return_value.write.return_value = None
        tempfile_mock.__enter__.return_value.seek.return_value = None
        tempfile_mock.__enter__.return_value.name = 'temp'

        network_mock = mock.MagicMock()
        network_mock.LoadScripts.return_value = None

        with mock.patch('tempfile.NamedTemporaryFile') as tempfile_mock_fn:
            tempfile_mock_fn.return_value = tempfile_mock
            motoroutput.light_up_led(network_mock, 0, 0, 0)
            tempfile_mock_fn.assert_called_once()
            tempfile_mock.__enter__.assert_called_once()  # enter context manager
            self.assertEqual(tempfile_mock.__enter__.return_value.write.call_count, 5)
            tempfile_mock.__enter__.return_value.seek.assert_called_once()
            network_mock.LoadScripts.assert_called_once()

    @mock.patch('visualswarm.control.motoroutput.distribute_overall_speed', return_value=(None, None))
    @mock.patch('numpy.random.uniform', return_value=1)
    def test_step_random_walk(self, mock_random, mock_distribute):
        with mock.patch('visualswarm.contrib.control.DPSI_MAX_EXP', 5):
            with mock.patch('visualswarm.contrib.control.V_EXP_RW', 10):
                [_, _] = motoroutput.step_random_walk()
                mock_random.assert_called_once_with(-5, 5, 1)
                mock_distribute.assert_called_once_with(10, 1)

    def test_hardlimit_motor_speed(self):
        with mock.patch('visualswarm.contrib.control.MAX_MOTOR_SPEED', 1):
            # preserving signs
            v_l, v_r = 2, 2
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [1, 1])
            v_l, v_r = 2, -2
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [1, -1])
            v_l, v_r = -2, 2
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [-1, 1])
            v_l, v_r = -2, -2
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [-1, -1])

            # preserving ratios
            v_l, v_r = 1, 2
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [0.5, 1])
            v_l, v_r = 2, 1
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [1, 0.5])

            # handling zeros
            v_l, v_r = 0, 2
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [0, 1])
            v_l, v_r = 0, -2
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [0, -1])
            v_l, v_r = 2, 0
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [1, 0])
            v_l, v_r = -2, 0
            self.assertEqual(motoroutput.hardlimit_motor_speed(v_l, v_r), [-1, 0])

    def test_distribute_overall_speed(self):
        with mock.patch('visualswarm.contrib.control.MOTOR_SCALE_CORRECTION', 1):
            with mock.patch('numpy.pi', 1):
                v = 100

                dpsi = -1 / 2  # as if multiplied with pi
                self.assertEqual(motoroutput.distribute_overall_speed(v, dpsi), [50, 150])

                dpsi = 1 / 2  # as if multiplied with pi
                self.assertEqual(motoroutput.distribute_overall_speed(v, dpsi), [150, 50])

                dpsi = 0
                self.assertEqual(motoroutput.distribute_overall_speed(v, dpsi), [100, 100])

                dpsi = -1  # as if multiplied with pi
                self.assertEqual(motoroutput.distribute_overall_speed(v, dpsi), [0, 200])

                dpsi = 1  # as if multiplied with pi
                self.assertEqual(motoroutput.distribute_overall_speed(v, dpsi), [200, 0])
