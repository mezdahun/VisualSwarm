"""
@author: mezdahun
@description: Testing on non Raspbian OS will cause errors when importing Raspbian HW specific
packages. We avoid this by faking the HW during testing.
"""

import sys
from unittest import mock

import fake_rpi


class FakePicameraArray(object):
    """faking picamera.array.PiRGBArray"""
    PiRGBArray = mock.MagicMock()

class FakePicameraException(object):
    """faking picamera.array.PiRGBArray"""
    PiCameraValueError = mock.MagicMock()

sys.modules['picamera'] = fake_rpi.picamera
sys.modules['picamera.array'] = FakePicameraArray
sys.modules['picamera.exc'] = FakePicameraException

# indicating flag when HW is faked
FAKE_STATUS = True
