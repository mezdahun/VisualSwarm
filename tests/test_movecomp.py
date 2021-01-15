from unittest import TestCase, mock

import numpy as np

from visualswarm.behavior import movecomp


class VAcquireTest(TestCase):

    def test_dPhi_V_of(self):
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

    @mock.patch('visualswarm.behavior.movecomp.dPhi_V_of')
    @mock.patch('scipy.integrate.trapz')
    def test_compute_control_params(self, mock_integrate, mock_dphi):
        mock_integrate.return_value = 100
        mock_dphi.return_value = np.zeros(10)
        phi = np.zeros(10)

        with mock.patch('visualswarm.contrib.flockparams.GAM', 2):
            with mock.patch('visualswarm.contrib.flockparams.V0', 5):
                vel_now = 10
                dv, dpsi = movecomp.compute_control_params(vel_now, phi, np.zeros(10))
                self.assertEqual(dv, 90)
                self.assertEqual(dpsi, 100)
