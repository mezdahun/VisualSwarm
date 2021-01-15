import datetime
from unittest import TestCase, mock

from freezegun import freeze_time

from visualswarm.monitoring import system_monitor


class SystemMonitorTest(TestCase):

    @freeze_time("2000-01-01")
    @mock.patch('visualswarm.env.EXIT_CONDITION', True)
    @mock.patch('time.sleep')
    @mock.patch('psutil.disk_usage')
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.cpu_percent')
    @mock.patch('psutil.sensors_temperatures')
    def test_system_monitor(self, fake_temp, fake_cpu_perc, fake_vmem, fake_disk_usage, fake_sleep):
        with mock.patch('visualswarm.monitoring.ifdb.create_ifclient') as fake_create_client:
            fake_ifclient = mock.MagicMock()
            fake_ifclient.write_points.return_value = None
            fake_create_client.return_value = fake_ifclient

            disk = mock.Mock(percent=10)
            # disk.percent = 10
            fake_disk_usage.return_value = disk

            mem = mock.Mock(percent=11)
            # mem.percent = 11
            fake_vmem.return_value = mem

            term = mock.Mock(current=16)
            # term.current = 16
            ret_dict = {'cpu_thermal': [term]}
            fake_temp.return_value = ret_dict

            fake_cpu_perc.return_value = [12, 13, 14, 15]

            ok_body = [
                {
                    "measurement": "system_parameters",
                    "time": datetime.datetime(2000, 1, 1),
                    "fields": {
                        "disk_percent": 10,
                        "mem_percent": 11,
                        "cpu_1": 12,
                        "cpu_2": 13,
                        "cpu_3": 14,
                        "cpu_4": 15,
                        "cpu_temperature": 16
                    }
                }
            ]

            fake_sleep.return_value = None
            system_monitor.system_monitor()
            fake_ifclient.write_points.assert_called_once_with(ok_body)
