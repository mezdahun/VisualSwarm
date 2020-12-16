from unittest import TestCase

from visualswarm import app


class AppTest(TestCase):

    def test_health(self):
        with self.assertLogs('visualswarm.app', level='INFO') as cm:
            app.health()
            self.assertEqual(cm.output, ['INFO:visualswarm.app:VisualSwarm application OK!'])

    @mock.patch('visualswarm.app.Process')
    @mock.patch('visualswarm.app.Queue')
    def test_start_vision_stream(self, mockQueue, mockProcess):
        if FAKE_STATUS:
            mp = mockProcess.return_value
            app.start_vision_stream()
            self.assertEqual(mp.start.call_count, 2)
            self.assertEqual(mp.join.call_count, 2)
            self.assertEqual(mockQueue.call_count, 2)
