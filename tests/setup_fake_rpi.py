"""
@author: mezdahun
@description: Testing on non Raspbian OS will cause errors when importing Raspbian HW specific
packages. We avoid this by faking the HW during testing.
"""

import sys

import fake_rpi

sys.modules['picamera'] = fake_rpi.picamera

# indicating flag when HW is faked
FAKE_STATUS = True
