from unittest import TestCase, mock

from dbus.exceptions import DBusException

from visualswarm.control import motorinterface


class MotorInterfaceTest(TestCase):

    def DBusException_raise(self, *args, **kwargs):
        raise DBusException

    def test_asebamedulla_health(self):
        network = mock.MagicMock()
        network.GetVariable.return_value = None

        # Case 1 : connection to thymio successful
        motorinterface.asebamedulla_health(network)
        network.GetVariable.assert_called_once_with("thymio-II", "acc", timeout=5)

        # Case 2 : DBusException is raised during communication
        network.GetVariable.reset_mock()
        network.GetVariable.side_effect = self.DBusException_raise
        self.assertEqual(motorinterface.asebamedulla_health(network), False)

    @mock.patch('os.system', return_value=None)
    @mock.patch('time.sleep', return_value=None)
    def test_asebamedulla_init(self, mock_sleep, mock_os):
        motorinterface.asebamedulla_init()
        mock_os.assert_called_once_with("(asebamedulla ser:name=Thymio-II &)")
        mock_sleep.assert_called_once_with(5)

    @mock.patch('os.system', return_value=None)
    def test_asebamedulla_end(self, mock_os):
        motorinterface.asebamedulla_end()
        mock_os.assert_called_once_with("pkill -f asebamedulla")
