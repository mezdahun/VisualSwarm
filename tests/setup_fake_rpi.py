"""
@author: mezdahun
@description: Testing on non Raspbian OS will cause errors when importing Raspbian HW specific
packages. We avoid this by faking the HW during testing.
"""

import sys

import fake_rpi


class fake_picamera_array(object):
    """faking picamera.array.PiRGBArray"""
    PiRGBArray = None


sys.modules['picamera'] = fake_rpi.picamera
sys.modules['picamera.array'] = fake_picamera_array

# indicating flag when HW is faked
FAKE_STATUS = True
