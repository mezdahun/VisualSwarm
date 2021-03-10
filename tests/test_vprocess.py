from unittest import TestCase, mock

import numpy as np
from freezegun import freeze_time

from visualswarm.vision import vprocess


class VProcessTest(TestCase):

    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    def test_high_level_vision(self):
        with mock.patch('cv2.cvtColor') as cvtColor:
            with mock.patch('cv2.inRange') as inRange:
                with mock.patch('cv2.GaussianBlur') as GaussianBlur:
                    with mock.patch('cv2.medianBlur') as medianBlur:
                        with mock.patch('cv2.findContours') as findContours:
                            with mock.patch('cv2.contourArea') as contourArea:
                                with mock.patch('cv2.convexHull') as convexHull:
                                    with mock.patch('cv2.drawContours') as drawContours:
                                        # Case 1 : no interactive parameter tuning
                                        with mock.patch('visualswarm.contrib.vision.FIND_COLOR_INTERACTIVE', False):
                                            cvtColor.return_value = None
                                            inRange.return_value = None
                                            GaussianBlur.return_value = None
                                            medianBlur.return_value = mock.MagicMock()
                                            medianBlur.copy.return_value = None
                                            findContours.return_value = ([None], None)
                                            contourArea.return_value = -1  # shall be smaller than any int threshold
                                            convexHull.return_value = None
                                            drawContours.return_value = None

                                            img = None
                                            frame_id = 15
                                            capture_timestamp = None
                                            raw_vision_stream = mock.MagicMock()
                                            raw_vision_stream.get.return_value = (img, frame_id, capture_timestamp)
                                            high_level_vision_stream = mock.MagicMock()
                                            high_level_vision_stream.put.return_value = None
                                            vprocess.high_level_vision(raw_vision_stream, high_level_vision_stream)

                                            self.assertEqual(cvtColor.call_count, 1)
                                            self.assertEqual(inRange.call_count, 1)
                                            self.assertEqual(GaussianBlur.call_count, 1)
                                            self.assertEqual(medianBlur.call_count, 1)
                                            self.assertEqual(findContours.call_count, 1)
                                            self.assertEqual(contourArea.call_count, 1)
                                            self.assertEqual(convexHull.call_count, 0)
                                            self.assertEqual(drawContours.call_count, 3)

                                        # Case 2 : Interactive Parameter tuning
                                        cvtColor.reset_mock()
                                        inRange.reset_mock()
                                        GaussianBlur.reset_mock()
                                        medianBlur.reset_mock()
                                        medianBlur.reset_mock()
                                        findContours.reset_mock()
                                        contourArea.reset_mock()
                                        convexHull.reset_mock()
                                        drawContours.reset_mock()
                                        with mock.patch('visualswarm.contrib.vision.FIND_COLOR_INTERACTIVE', True):
                                            with mock.patch('visualswarm.contrib.vision.MIN_BLOB_AREA', 0):
                                                cvtColor.return_value = [[[0]]]
                                                inRange.return_value = None
                                                GaussianBlur.return_value = None
                                                medianBlur.return_value = mock.MagicMock()
                                                medianBlur.copy.return_value = None
                                                findContours.return_value = ([None], None)
                                                contourArea.return_value = 10  # now larger than threshold
                                                convexHull.return_value = None
                                                drawContours.return_value = None

                                                img = None
                                                frame_id = 15
                                                capture_timestamp = None
                                                raw_vision_stream = mock.MagicMock()
                                                raw_vision_stream.get.return_value = (img, frame_id, capture_timestamp)
                                                high_level_vision_stream = mock.MagicMock()
                                                high_level_vision_stream.put.return_value = None
                                                parameter_stream = mock.MagicMock()
                                                parameter_stream.get.return_value = (0, 0, 0, 0, 0, 0)
                                                parameter_stream.qsize.return_value = 2
                                                visualization_stream = mock.MagicMock()
                                                visualization_stream.put.return_value = None
                                                vprocess.high_level_vision(raw_vision_stream,
                                                                           high_level_vision_stream,
                                                                           visualization_stream,
                                                                           parameter_stream)

                                                self.assertEqual(parameter_stream.get.call_count, 1)
                                                self.assertEqual(cvtColor.call_count, 2)
                                                self.assertEqual(inRange.call_count, 1)
                                                self.assertEqual(GaussianBlur.call_count, 1)
                                                self.assertEqual(medianBlur.call_count, 1)
                                                self.assertEqual(findContours.call_count, 1)
                                                self.assertEqual(contourArea.call_count, 1)
                                                self.assertEqual(convexHull.call_count, 1)
                                                self.assertEqual(drawContours.call_count, 3)
                                                self.assertEqual(visualization_stream.put.call_count, 1)

    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    def test_visualizer(self):
        # Case 1 no visualization
        with self.assertLogs('visualswarm.app', level='INFO') as cm:
            vprocess.visualizer(None)
            self.assertEqual(cm.output,
                             ['INFO:visualswarm.app:Visualization stream is None, visualization process returns!'])
        # Case 2 visualization
        # Case 2.a no interactive plotting
        with mock.patch('visualswarm.contrib.vision.FIND_COLOR_INTERACTIVE', False):
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

        # Case 2.b with interactive plotting
        with mock.patch('visualswarm.contrib.vision.FIND_COLOR_INTERACTIVE', True):
            with mock.patch('cv2.imshow') as fake_imshow:
                with mock.patch('cv2.resize') as fake_resize:
                    with mock.patch('cv2.waitKey') as fake_waitKey:
                        with mock.patch('cv2.namedWindow') as fake_namedWindow:
                            with mock.patch('cv2.createTrackbar') as fake_createTrackbar:
                                with mock.patch('cv2.getTrackbarPos') as fake_getTrackbarPos:
                                    fake_imshow.return_value = None
                                    fake_resize.return_value = None
                                    fake_waitKey.return_value = None
                                    fake_namedWindow.return_value = None
                                    fake_createTrackbar.return_value = None
                                    fake_getTrackbarPos.return_value = 0
                                    img = None
                                    mask = None
                                    frame_id = 15
                                    visualizer_stream = mock.MagicMock()
                                    visualizer_stream.get.return_value = (img, mask, frame_id)
                                    parameter_stream = mock.MagicMock()
                                    parameter_stream.put.return_value = None
                                    vprocess.visualizer(visualizer_stream, parameter_stream)
                                    self.assertEqual(fake_imshow.call_count, 3)
                                    self.assertEqual(fake_resize.call_count, 2)
                                    self.assertEqual(fake_waitKey.call_count, 1)
                                    self.assertEqual(fake_createTrackbar.call_count, 6)
                                    self.assertEqual(fake_namedWindow.call_count, 1)
                                    self.assertEqual(parameter_stream.put.call_count, 1)

    @freeze_time("2000-01-01")
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    def test_VPF_extraction(self):
        with mock.patch('visualswarm.monitoring.ifdb.create_ifclient') as fake_create_client:
            fake_ifclient = mock.MagicMock()
            fake_ifclient.write_points.return_value = None
            fake_create_client.return_value = fake_ifclient

            with mock.patch('visualswarm.contrib.projection.H_MARGIN', 0):
                with mock.patch('visualswarm.contrib.projection.W_MARGIN', 0):
                    with mock.patch('numpy.max') as fake_npmax:
                        # Case 1 : no saving to db
                        with mock.patch('visualswarm.contrib.monitorparams.SAVE_PROJECTION_FIELD', False):
                            fake_npmax.return_value = np.array([1, 2, 3])
                            img = None
                            mask = np.array([[1, 2, 3], [0, 0, 0]])
                            frame_id = 15
                            capture_timestamp = None
                            vision_stream = mock.MagicMock()
                            vision_stream.get.return_value = (img, mask, frame_id, capture_timestamp)
                            VPF_stream = mock.MagicMock()
                            VPF_stream.put.return_value = None
                            vprocess.VPF_extraction(vision_stream, VPF_stream)
                            fake_npmax.assert_called_once()

                        # Case 2 : saving to db
                        fake_npmax.reset_mock()
                        with mock.patch('visualswarm.contrib.monitorparams.SAVE_PROJECTION_FIELD', True):
                            with mock.patch('visualswarm.contrib.monitorparams.DOWNGRADING_FACTOR', 1):
                                with mock.patch('visualswarm.monitoring.ifdb.pad_to_n_digits') as fake_pad:
                                    with mock.patch('numpy.max') as fake_npmax:
                                        fake_npmax.return_value = np.array([1, 2, 3])
                                        fake_pad.return_value = '001'
                                        img = None
                                        mask = np.array([[1, 2, 3], [0, 0, 0]])
                                        frame_id = 15
                                        capture_timestamp = None
                                        vision_stream = mock.MagicMock()
                                        vision_stream.get.return_value = (img, mask, frame_id, capture_timestamp)
                                        VPF_stream = mock.MagicMock()
                                        VPF_stream.put.return_value = None
                                        vprocess.VPF_extraction(vision_stream, VPF_stream)
                                        fake_npmax.assert_called_once()

    def test_nothing(self):
        self.assertEqual(vprocess.nothing(None), None)
