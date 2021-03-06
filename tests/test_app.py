from unittest import TestCase, mock

from setup_fake_rpi import FAKE_STATUS

import visualswarm.contrib.vision
from visualswarm import app


class AppTest(TestCase):

    def test_health(self):
        with self.assertLogs('visualswarm.app', level='INFO') as cm:
            app.health()
            self.assertEqual(cm.output, ['INFO:visualswarm.app:VisualSwarm application OK!'])

    @mock.patch('visualswarm.app.Process')
    @mock.patch('visualswarm.app.Queue')
    @mock.patch('visualswarm.control.motorinterface.asebamedulla_init', return_value=None)
    def test_start_application(self, mock_asebamedulla_init, mockQueue, mockProcess):
        if FAKE_STATUS:
            with mock.patch('visualswarm.env.INFLUX_FRESH_DB_UPON_START', False):
                # Case 1 with interactive visualization
                num_processes = 7 + visualswarm.contrib.vision.NUM_SEGMENTATION_PROCS
                with mock.patch('visualswarm.contrib.vision.FIND_COLOR_INTERACTIVE', True):
                    # Case 1/A visualization started even if didnt set in env
                    with mock.patch('visualswarm.contrib.vision.SHOW_VISION_STREAMS', False):
                        num_queues = 8
                        mp = mockProcess.return_value
                        app.start_application(with_control=True)
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)
                        mock_asebamedulla_init.assert_called_once()

                    # Case 1/B visualization was desired in env
                    mockProcess.reset_mock()
                    mockQueue.reset_mock()
                    mock_asebamedulla_init.reset_mock()
                    with mock.patch('visualswarm.contrib.vision.SHOW_VISION_STREAMS', True):
                        num_queues = 9
                        mp = mockProcess.return_value
                        app.start_application(with_control=True)
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)
                        mock_asebamedulla_init.assert_called_once()

                with mock.patch('visualswarm.contrib.vision.FIND_COLOR_INTERACTIVE', False):
                    # Case 2 with no visualization at all
                    mockProcess.reset_mock()
                    mockQueue.reset_mock()
                    mock_asebamedulla_init.reset_mock()
                    with mock.patch('visualswarm.contrib.vision.SHOW_VISION_STREAMS', False):
                        num_queues = 6
                        mp = mockProcess.return_value
                        app.start_application(with_control=True)
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)
                        mock_asebamedulla_init.assert_called_once()

                    # Case 2/B visualization but not interactive
                    mockProcess.reset_mock()
                    mockQueue.reset_mock()
                    mock_asebamedulla_init.reset_mock()
                    with mock.patch('visualswarm.contrib.vision.SHOW_VISION_STREAMS', True):
                        num_queues = 7
                        mp = mockProcess.return_value
                        app.start_application(with_control=True)
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)
                        mock_asebamedulla_init.assert_called_once()

                # Case 3 starting with DB wipe
                mock_asebamedulla_init.reset_mock()
                with mock.patch('visualswarm.contrib.vision.FIND_COLOR_INTERACTIVE', False):
                    with mock.patch('visualswarm.env.INFLUX_FRESH_DB_UPON_START', True):
                        with mock.patch('visualswarm.monitoring.ifdb.create_ifclient') as fake_create_client:
                            fake_ifclient = mock.MagicMock()
                            fake_ifclient.drop_database.return_value = None
                            fake_ifclient.create_database.return_value = None
                            fake_create_client.return_value = fake_ifclient
                            app.start_application(with_control=True)
                            fake_create_client.assert_called_once()
                            fake_ifclient.drop_database.assert_called_once()
                            fake_ifclient.create_database.assert_called_once()
                            mock_asebamedulla_init.assert_called_once()

    @mock.patch('visualswarm.app.start_application', return_value=None)
    def test_start_application_with_control(self, mock_start):
        app.start_application_with_control()
        mock_start.assert_called_once_with(with_control=True)
