from unittest import TestCase, mock

import numpy as np

from visualswarm.behavior import movecomp


class VAcquireTest(TestCase):

    @mock.patch('visualswarm.vision.vacquire.PiRGBArray')
    @mock.patch('picamera.PiCamera.capture_continuous', create=True)
    @mock.patch('visualswarm.vision.vacquire.stabilize_color_space_params')
    def test_dPhi_V_of(self, mock_stabilize, mock_PiC_loop, mock_PiRGBArray):
        phi = np.linspace(0, 9, 10)

        # Case 1: object in FOV not touching edge
        V = np.zeros(10)
        V[3:5] = 1
        vdiff_ok = np.zeros(10)
        vdiff_ok[2] = 1
        vdiff_ok[4] = -1
        vdiff = movecomp.dPhi_V_of(phi, V)
        np.testing.assert_array_equal(vdiff, vdiff_ok)

        # Case 2: object in FOV touching right edge
        V = np.zeros(10)
        V[7::] = 1
        vdiff_ok = np.zeros(10)
        vdiff_ok[6] = 1
        vdiff_ok[-1] = -1
        vdiff = movecomp.dPhi_V_of(phi, V)
        np.testing.assert_array_equal(vdiff, vdiff_ok)

        # Case 3: object in FOV touching left edge
        V = np.zeros(10)
        V[0:5] = 1
        vdiff_ok = np.zeros(10)
        vdiff_ok[0] = 1
        vdiff_ok[5] = -1
        vdiff = movecomp.dPhi_V_of(phi, V)
        np.testing.assert_array_equal(vdiff, vdiff_ok)

        # Case 3: object circularly goes through on edges
        V = np.zeros(10)
        V[0:2] = 1
        V[8::] = 1
        vdiff_ok = np.zeros(10)
        vdiff_ok[1] = -1
        vdiff_ok[7] = 1
        vdiff = movecomp.dPhi_V_of(phi, V)
        np.testing.assert_array_equal(vdiff, vdiff_ok)
