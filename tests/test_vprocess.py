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

