from fabric import Connection
from time import sleep

from visualswarm.contrib import puppetmaster
# Imports the Cloud Logging client library
import google.cloud.logging
from google.cloud.logging import Resource

# Instantiates a client
client = google.cloud.logging.Client()

# Retrieves a Cloud Logging handler based on the environment
# you're running in and integrates the handler with the
# Python logging module. By default this captures all logs
# at INFO level and higher
client.get_default_handler()
client.setup_logging()

import logging
# # setup logging
# logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('ERROR')

def start_swarm():
    logger.info('Puppetmaster started!')
    # with Connection(puppetmaster.HOSTS[0], user=puppetmaster.UNAME) as c:
    #     c.connect_kwargs.password = puppetmaster.PSWD
    #     c.run('cd Desktop/VisualSwarm && dtach -n /tmp/tmpdtach pipenv run vswrm-start-vision', hide=True, pty=False)
    #     start_result = c.run('ps ax  | grep "dtach -n /tmp/tmpdtach pipenv run vswrm-start-vision"')
    #     PID = start_result.stdout.split()[0]
    #     print(f'Started process with PID: {int(PID)}')
    #     print('going to sleeeeeep...')
    #     sleep(5)
    #     print('Waking up and killing process!')
    #     end_result = c.run(f'kill -INT {int(PID)}')
    #     print(end_result.stdout)