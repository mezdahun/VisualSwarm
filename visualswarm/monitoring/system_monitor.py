"""
@author: mezdahun
@description: System Monitoring
@source: https://simonhearne.com/2020/pi-metrics-influx/
"""
import datetime
from time import sleep
import psutil
from visualswarm.monitoring import ifdb


def system_monitor():
    measurement_name = "system_parameters"
    ifclient = ifdb.create_ifclient()

    while True:
        # take a timestamp for this measurement
        time = datetime.datetime.utcnow()

        # collect some stats from psutil
        disk = psutil.disk_usage('/')
        mem = psutil.virtual_memory()
        load = psutil.getloadavg()
        cpu_percent = psutil.cpu_percent(percpu=True)

        # format the data as a single measurement for influx
        body = [
            {
                "measurement": measurement_name,
                "time": time,
                "fields": {
                    "load_1": load[0],
                    "load_5": load[1],
                    "load_15": load[2],
                    "disk_percent": disk.percent,
                    "disk_free": disk.free,
                    "disk_used": disk.used,
                    "mem_percent": mem.percent,
                    "mem_free": mem.free,
                    "mem_used": mem.used,
                    "cpu_1": cpu_percent[0],
                    "cpu_2": cpu_percent[1],
                    "cpu_3": cpu_percent[2],
                    "cpu_4": cpu_percent[3]
                }
            }
        ]

        # write the measurement
        ifclient.write_points(body)

        sleep(0.25)
