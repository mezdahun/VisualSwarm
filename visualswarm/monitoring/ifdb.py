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