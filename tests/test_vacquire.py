from unittest import TestCase, mock

from setup_fake_rpi import FAKE_STATUS

from visualswarm.vision import vacquire


class VAcquireTest(TestCase):

    @mock.patch('visualswarm.vision.vacquire.PiRGBArray')
    @mock.patch('picamera.PiCamera.capture_continuous', create=True)
    def test_raw_vision(self, mock_PiC_loop, mock_PiRGBArray):
        if FAKE_STATUS:
            # under faking HW
            mock_PiRGBArray.return_value = mock.MagicMock()

            frame = mock.MagicMock
            frame.array = [0, 0, 0]
            mock_PiC_loop.return_value = [frame]

            raw_vision_stream = mock.MagicMock()
            raw_vision_stream.put.return_value = None

            vacquire.raw_vision(raw_vision_stream)

            mock_PiRGBArray.assert_called_once()
            raw_vision_stream.put.assert_called_once()
            array_instance = mock_PiRGBArray()
            array_instance.truncate.assert_called_once()
