from unittest import TestCase, mock

from visualswarm import env
from visualswarm.monitoring import ifdb


class IFDB(TestCase):

    def test_pad_to_n_digits(self):
        self.assertEqual(ifdb.pad_to_n_digits(2, 5), '00002')
        self.assertEqual(ifdb.pad_to_n_digits(2), '002')
        self.assertEqual(ifdb.pad_to_n_digits(32), '032')
        self.assertEqual(ifdb.pad_to_n_digits(132), '132')
        self.assertEqual(ifdb.pad_to_n_digits(1132), '1132')

    @mock.patch('visualswarm.monitoring.ifdb.InfluxDBClient')
    def test_create_ifclient(self, mock_IDBClient):
        mock_IDBClient.return_value = mock.MagicMock()
        ifdb.create_ifclient()
        mock_IDBClient.assert_called_once_with(env.INFLUX_HOST,
                                               env.INFLUX_PORT,
                                               env.INFLUX_USER,
                                               env.INFLUX_PSWD,
                                               env.INFLUX_DB_NAME)
