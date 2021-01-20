from unittest import TestCase, mock

from setup_fake_rpi import FAKE_STATUS

from visualswarm import app
from visualswarm.contrib import segmentation


class AppTest(TestCase):

    def test_health(self):
        with self.assertLogs('visualswarm.app', level='INFO') as cm:
            app.health()
            self.assertEqual(cm.output, ['INFO:visualswarm.app:VisualSwarm application OK!'])

    @mock.patch('visualswarm.app.Process')
    @mock.patch('visualswarm.app.Queue')
    def test_start_vision_stream(self, mockQueue, mockProcess):
        if FAKE_STATUS:
            with mock.patch('visualswarm.env.INFLUX_FRESH_DB_UPON_START', False):
                # Case 1 with interactive visualization
                num_processes = 5 + segmentation.NUM_SEGMENTATION_PROCS
                with mock.patch('visualswarm.contrib.visual.FIND_COLOR_INTERACTIVE', True):
                    # Case 1/A visualization started even if didnt set in env
                    with mock.patch('visualswarm.contrib.visual.SHOW_VISION_STREAMS', False):
                        num_queues = 6
                        mp = mockProcess.return_value
                        app.start_vision_stream()
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)

                    # Case 1/B visualization was desired in env
                    mockProcess.reset_mock()
                    mockQueue.reset_mock()
                    with mock.patch('visualswarm.contrib.visual.SHOW_VISION_STREAMS', True):
                        num_queues = 7
                        mp = mockProcess.return_value
                        app.start_vision_stream()
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)

                with mock.patch('visualswarm.contrib.visual.FIND_COLOR_INTERACTIVE', False):
                    # Case 2 with no visualization at all
                    mockProcess.reset_mock()
                    mockQueue.reset_mock()
                    with mock.patch('visualswarm.contrib.visual.SHOW_VISION_STREAMS', False):
                        num_queues = 4
                        mp = mockProcess.return_value
                        app.start_vision_stream()
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)

                    # Case 2/B visualization but not interactive
                    mockProcess.reset_mock()
                    mockQueue.reset_mock()
                    with mock.patch('visualswarm.contrib.visual.SHOW_VISION_STREAMS', True):
                        num_queues = 5
                        mp = mockProcess.return_value
                        app.start_vision_stream()
                        self.assertEqual(mp.start.call_count, num_processes)
                        self.assertEqual(mp.join.call_count, num_processes)
                        self.assertEqual(mockQueue.call_count, num_queues)

                # Case 3 starting with DB wipe
                with mock.patch('visualswarm.contrib.visual.FIND_COLOR_INTERACTIVE', False):
                    with mock.patch('visualswarm.env.INFLUX_FRESH_DB_UPON_START', True):
                        with mock.patch('visualswarm.monitoring.ifdb.create_ifclient') as fake_create_client:
                            fake_ifclient = mock.MagicMock()
                            fake_ifclient.drop_database.return_value = None
                            fake_ifclient.create_database.return_value = None
                            fake_create_client.return_value = fake_ifclient
                            app.start_vision_stream()
                            fake_create_client.assert_called_once()
                            fake_ifclient.drop_database.assert_called_once()
                            fake_ifclient.create_database.assert_called_once()
