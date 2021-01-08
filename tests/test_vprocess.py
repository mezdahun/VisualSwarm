from unittest import TestCase, mock, skip

from visualswarm.vision import vprocess


class VProcessTest(TestCase):

    @skip
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    def test_high_level_vision(self):
        img = None
        frame_id = 15
        raw_vision_stream = mock.MagicMock()
        raw_vision_stream.get.return_value = (img, frame_id)
        vprocess.high_level_vision(raw_vision_stream, None)

    @skip
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    def test_visualizer(self):
        # Case 1 no visualization
        with self.assertLogs('visualswarm.app', level='INFO') as cm:
            vprocess.visualizer(None)
            self.assertEqual(cm.output,
                             ['INFO:visualswarm.app:Visualization stream is None, visualization process returns!'])

        # Case 2 visualization
        # Case 2.a no interactive plotting
        with mock.patch('visualswarm.contrib.visual.FIND_COLOR_INTERACTIVE', False):
            with mock.patch('cv2.imshow') as fake_imshow:
                with mock.patch('cv2.resize') as fake_resize:
                    with mock.patch('cv2.waitKey') as fake_waitKey:
                        fake_imshow.return_value = None
                        fake_resize.return_value = None
                        fake_waitKey.return_value = None
                        img = None
                        mask = None
                        frame_id = 15
                        visualizer_stream = mock.MagicMock()
                        visualizer_stream.get.return_value = (img, mask, frame_id)
                        vprocess.visualizer(visualizer_stream)