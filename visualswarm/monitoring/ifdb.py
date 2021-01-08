"""
@author: mezdahun
@description: Helper functions for InfluxDB
"""
from influxdb import InfluxDBClient

from visualswarm import env


def create_ifclient():
    """Connecting to the InfluxDB defined with environmental variables and returning a client instance.
        Args:
            None
        Returns:
            ifclient: InfluxDBClient connected to the database defined in the environment variables."""
    ifclient = InfluxDBClient(env.INFLUX_HOST,
                              env.INFLUX_PORT,
                              env.INFLUX_USER,
                              env.INFLUX_PSWD,
                              env.INFLUX_DB_NAME)
    return ifclient


def pad_to_n_digits(number, n=3):
    """
    Padding a single number to n digits with leading zeros so that lexicographic sorting does not mix fields of a
    measurement in InfluxDb.
        Args:
            number: int or string of a number
            n: the number of desired digits of the output
        Returns:
            padded number or the input number if it already has the desired length
    """
    len_diff = n - len(str(number))
    if len_diff > 0:
        return len_diff * '0' + str(number)
    else:
        return str(number)
