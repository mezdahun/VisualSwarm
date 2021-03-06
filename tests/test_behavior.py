import datetime
from unittest import TestCase, mock

import numpy as np
from freezegun import freeze_time

from visualswarm.behavior import behavior


class MoveCompTest(TestCase):

    @freeze_time("2000-01-01")
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    def test_VPF_to_behavior(self):
        with mock.patch('visualswarm.monitoring.ifdb.create_ifclient') as fake_create_client:
            with mock.patch('visualswarm.contrib.vision.PHI_START', -np.pi):
                with mock.patch('visualswarm.contrib.vision.PHI_END', np.pi):
                    with mock.patch('visualswarm.behavior.statevarcomp.compute_state_variables') as fake_control_params:

                        # Mocking calculations
                        fake_ifclient = mock.MagicMock()
                        fake_ifclient.write_points.return_value = None
                        fake_create_client.return_value = fake_ifclient

                        fake_control_params.return_value = (1, 1)

                        # Mocking input stream
                        p_field = np.zeros(10)
                        VPF_stream = mock.MagicMock()
                        capture_timestamp = datetime.datetime(2000, 1, 1)
                        VPF_stream.get.return_value = (p_field, capture_timestamp)

                        # Mocking output streams
                        control_stream = mock.MagicMock()
                        control_stream.put.return_value = None
                        movement_mode_stream = mock.MagicMock()
                        movement_mode_stream.put.return_value = None

                        # Case 1: no save control params
                        with mock.patch('visualswarm.contrib.monitoring.SAVE_CONTROL_PARAMS', False):
                            behavior.VPF_to_behavior(VPF_stream, control_stream, movement_mode_stream, False)
                            fake_create_client.assert_called_once()
                            self.assertEqual(fake_control_params.call_count, 1)

                        # resetting mocks
                        fake_create_client.reset_mock()
                        fake_control_params.reset_mock()

                        # Mocking calculations
                        fake_ifclient = mock.MagicMock()
                        fake_ifclient.write_points.return_value = None
                        fake_create_client.return_value = fake_ifclient

                        fake_control_params.return_value = (1, 1)

                        # Case 2: save control params to ifdb
                        with mock.patch('visualswarm.contrib.monitoring.SAVE_CONTROL_PARAMS', True):
                            behavior.VPF_to_behavior(VPF_stream, control_stream, movement_mode_stream, False)
                            fake_create_client.assert_called_once()
                            self.assertEqual(fake_control_params.call_count, 1)
                            fake_ifclient.write_points.assert_called_once()

                        # Case 3: motor output turned off
                        behavior.VPF_to_behavior(VPF_stream, control_stream, movement_mode_stream, True)
                        control_stream.put.assert_called_once()
                        movement_mode_stream.put.assert_called_once()
