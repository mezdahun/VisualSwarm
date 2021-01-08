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
            num_processes = 3 + segmentation.NUM_SEGMENTATION_PROCS
            num_queues = 3
            mp = mockProcess.return_value
            app.start_vision_stream()
            self.assertEqual(mp.start.call_count, num_processes)
            self.assertEqual(mp.join.call_count, num_processes)
            self.assertEqual(mockQueue.call_count, num_queues)
