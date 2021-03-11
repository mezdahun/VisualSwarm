"""
@author: mezdahun
@description: System Monitoring
@source: https://simonhearne.com/2020/pi-metrics-influx/
"""
import datetime
from time import sleep
import psutil
from visualswarm.monitoring import ifdb
from visualswarm import env


def system_monitor():
    """
        Method to collect system rescource parameters and write them into influxDB instance defined according
        to visualswarm.monitoring.ifdb
    """
    measurement_name = "system_parameters"
    ifclient = ifdb.create_ifclient()

    while True:
        # take a timestamp for this measurement
        time = datetime.datetime.utcnow()

        # collect some stats from psutil
        disk = psutil.disk_usage('/')
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(percpu=True)

        cpu_temp = psutil.sensors_temperatures()['cpu_thermal'][0].current

        # format the data as a single measurement for influx
        body = [
            {
                "measurement": measurement_name,
                "time": time,
                "fields": {
                    "disk_percent": disk.percent,
                    "mem_percent": mem.percent,
                    "cpu_1": cpu_percent[0],
                    "cpu_2": cpu_percent[1],
                    "cpu_3": cpu_percent[2],
                    "cpu_4": cpu_percent[3],
                    "cpu_temperature": cpu_temp
                }
            }
        ]

        # write the measurement
        ifclient.write_points(body)

        sleep(0.25)

        # To test infinite loops
        if env.EXIT_CONDITION:
            break
