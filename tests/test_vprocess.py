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
                        self.assertEqual(fake_imshow.call_count, 2)
                        self.assertEqual(fake_resize.call_count, 2)
                        self.assertEqual(fake_waitKey.call_count, 1)

        # Case 2.b no interactive plotting
        with mock.patch('visualswarm.contrib.visual.FIND_COLOR_INTERACTIVE', True):
            with mock.patch('cv2.imshow') as fake_imshow:
                with mock.patch('cv2.resize') as fake_resize:
                    with mock.patch('cv2.waitKey') as fake_waitKey:
                        with mock.patch('cv2.namedWindow') as fake_namedWindow:
                            with mock.patch('cv2.createTrackbar') as fake_createTrackbar:
                                fake_imshow.return_value = None
                                fake_resize.return_value = None
                                fake_waitKey.return_value = None
                                fake_namedWindow.return_value = None
                                fake_createTrackbar.return_value = None
                                img = None
                                mask = None
                                frame_id = 15
                                visualizer_stream = mock.MagicMock()
                                visualizer_stream.get.return_value = (img, mask, frame_id)
                                vprocess.visualizer(visualizer_stream)
                                self.assertEqual(fake_imshow.call_count, 2)
                                self.assertEqual(fake_resize.call_count, 2)
                                self.assertEqual(fake_waitKey.call_count, 1)
                                self.assertEqual(fake_createTrackbar.call_count, 6)
                                self.assertEqual(fake_namedWindow.call_count, 1)
