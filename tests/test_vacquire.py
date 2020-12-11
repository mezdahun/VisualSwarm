from unittest import TestCase, mock

from setup_fake_rpi import FAKE_STATUS

from visualswarm.vision import vacquire


class VAcquireTest(TestCase):

    @mock.patch('picamera.PiCamera.start_preview', create=True)
    @mock.patch('picamera.PiCamera.stop_preview', create=True)
    def test_acq_image(self, mock_PiC_stop, mock_PiC_start):
        if FAKE_STATUS:
            # under faking HW
            mock_PiC_start.return_value = None
            mock_PiC_stop.return_value = None
            vacquire.acq_image()
            mock_PiC_start.assert_called_once()
            mock_PiC_stop.assert_called_once()
