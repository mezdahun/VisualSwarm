from unittest import TestCase, mock

from visualswarm.control import motoroutput


class MotorInterfaceTest(TestCase):

    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    @mock.patch('dbus.mainloop.glib.DBusGMainLoop', return_value=None)
    @mock.patch('dbus.SessionBus')
    @mock.patch('dbus.Interface')
    def test_control_thymio(self, mock_network_init, mock_dbus_sessionbus, mock_dbus_init):
        mock_network = mock.MagicMock(return_value=None)
        mock_network.SetVariable.return_value = None
        mock_network_init.return_value = mock_network

        mock_dbus_sessionbus.return_value.get_object.return_value = None

        # mocking streams
        control_stream = mock.MagicMock()
        control_stream.get.return_value = (1, 1)
        movement_mode_stream = mock.MagicMock()
        movement_mode_stream.get.return_value = "BEHAVE"

        # Case 1 : healthy connection via asebamedulla
        with mock.patch('visualswarm.control.motorinterface.asebamedulla_health') as mock_health:
            mock_health.return_value = True
            # Case 1/a : with no control
            motoroutput.control_thymio(control_stream, movement_mode_stream, with_control=False)
            control_stream.get.assert_called_once()
            movement_mode_stream.get.assert_called_once()

            control_stream.get.reset_mock()
            movement_mode_stream.get.reset_mock()
            mock_health.reset_mock()
            # Case 1/b : with control
            motoroutput.control_thymio(control_stream, movement_mode_stream, with_control=True)
            mock_dbus_init.assert_called_once_with(set_as_default=True)
            mock_dbus_sessionbus.assert_called_once()
            mock_network_init.assert_called_once()
            mock_health.assert_called_once()
            control_stream.get.assert_called_once()
            movement_mode_stream.get.assert_called_once()
            self.assertEqual(mock_network.SetVariable.call_count, 2)

            # Case 1/c: unknown movement mode
            movement_mode_stream = mock.MagicMock()
            movement_mode_stream.get.return_value = "UNKNOWN"
            with self.assertRaises(KeyboardInterrupt):
                motoroutput.control_thymio(control_stream, movement_mode_stream, with_control=True)

        # Case 2 : unhealthy connection via asebamedulla
        with mock.patch('visualswarm.control.motorinterface.asebamedulla_health') as mock_health:
            with mock.patch('visualswarm.control.motorinterface.asebamedulla_end'):
                mock_health.return_value = False
                control_stream.reset_mock()
                movement_mode_stream.get.reset_mock()
                motoroutput.control_thymio(control_stream, movement_mode_stream, with_control=False)
                control_stream.get.assert_called_once()
                movement_mode_stream.get.assert_called_once()

                with self.assertRaises(Exception):
                    motoroutput.control_thymio(control_stream, movement_mode_stream, with_control=True)

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
