from unittest import TestCase, mock

from setup_fake_rpi import FAKE_STATUS

from visualswarm.vision import vacquire


class VAcquireTest(TestCase):

    @mock.patch('visualswarm.vision.vacquire.PiRGBArray')
    @mock.patch('picamera.PiCamera.capture_continuous', create=True)
    @mock.patch('visualswarm.vision.vacquire.stabilize_color_space_params')
    def test_raw_vision(self, mock_stabilize, mock_PiC_loop, mock_PiRGBArray):
        if FAKE_STATUS:
            # under faking HW
            mock_stabilize.return_value = None

            mock_PiRGBArray.return_value = mock.MagicMock()

            frame = mock.MagicMock
            frame.array = [0, 0, 0]
            mock_PiC_loop.return_value = [frame]

            raw_vision_stream = mock.MagicMock()
            raw_vision_stream.put.return_value = None

            vacquire.raw_vision(raw_vision_stream)

            mock_stabilize.assert_called_once()
            mock_PiRGBArray.assert_called_once()
            raw_vision_stream.put.assert_called_once()
            array_instance = mock_PiRGBArray()
            array_instance.truncate.assert_called_once()

    @mock.patch('time.sleep')
    def test_stabilize_color_space_params(self, fake_sleep):
        fake_sleep.return_value = None
        picam = mock.MagicMock()
        picam.iso = 0
        picam.exposure_speed = 100
        picam.shutter_speed = 0
        picam.exposure_mode = 'on'
        picam.awb_gains = 'should remain'
        picam.awb_mode = 'on'

        # TODO fix hardcoded values
        vacquire.stabilize_color_space_params(picam)
        self.assertEqual(picam.iso, 300)
        self.assertEqual(picam.shutter_speed, 100)
        self.assertEqual(picam.exposure_mode, 'off')
        self.assertEqual(picam.awb_mode, 'off')
        self.assertEqual(picam.awb_gains, 'should remain')
