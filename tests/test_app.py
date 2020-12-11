from unittest import TestCase

from visualswarm import app


class AppTest(TestCase):

    def test_health(self):
        with self.assertLogs('visualswarm.app', level='INFO') as cm:
            app.health()
            self.assertEqual(cm.output, ['INFO:visualswarm.app:VisualSwarm application OK!'])
