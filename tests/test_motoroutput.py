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
        control_stream = mock.MagicMock()
        control_stream.get.return_value = (1, 1)

        # Case 1 : healthy connection via asebamedulla
        with mock.patch('visualswarm.control.motorinterface.asebamedulla_health') as mock_health:
            mock_health.return_value = True
            # Case 1/a : with no control
            motoroutput.control_thymio(control_stream, with_control=False)
            control_stream.get.assert_called_once()

            control_stream.get.reset_mock()
            mock_health.reset_mock()
            # Case 1/b : with control
            motoroutput.control_thymio(control_stream, with_control=True)
            mock_dbus_init.assert_called_once_with(set_as_default=True)
            mock_dbus_sessionbus.assert_called_once()
            mock_network_init.assert_called_once()
            mock_health.assert_called_once()
            control_stream.get.assert_called_once()
            self.assertEqual(mock_network.SetVariable.call_count, 2)

        # Case 2 : unhealthy connection via asebamedulla
        with mock.patch('visualswarm.control.motorinterface.asebamedulla_health') as mock_health:
            with mock.patch('visualswarm.control.motorinterface.asebamedulla_end'):
                mock_health.return_value = False
                control_stream.reset_mock()
                motoroutput.control_thymio(control_stream, with_control=False)
                control_stream.get.assert_called_once()

                with self.assertRaises(Exception):
                    motoroutput.control_thymio(control_stream, with_control=True)
