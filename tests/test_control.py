from unittest import TestCase, mock, skip

import numpy as np

# from visualswarm.behavior import control


class MoveCompTest(TestCase):

    @skip
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    def test_VPF_to_behavior(self):
        with mock.patch('visualswarm.monitoring.ifdb.create_ifclient') as fake_create_client:
            fake_ifclient = mock.MagicMock()
            fake_ifclient.write_points.return_value = None
            fake_create_client.return_value = fake_ifclient

            p_field = np.zeros(10)
            VPF_stream = mock.MagicMock()
            capture_timestamp = None
            VPF_stream.get.return_value = (p_field, capture_timestamp)
