from fabric import Connection
from time import sleep

from visualswarm.contrib import puppetmaster
# Imports the Cloud Logging client library

import logging
# # setup logging
# logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

def reinstall_robots():
    logger.info("Updating robots' virtual environment...")
    with Connection(list(puppetmaster.HOSTS.values())[0], user=puppetmaster.UNAME) as c:
        c.connect_kwargs.password = puppetmaster.PSWD
        result = c.run('cd Desktop/VisualSwarm && '
                       'git pull && '
                       'pipenv --rm && '
                       'pipenv install -d --skip-lock -e .',
                       hide=True,
                       pty=False)
        print(result.stdout)

def update_robots():
    logger.info("Updating robots' virtual environment...")
    with Connection(list(puppetmaster.HOSTS.values())[0], user=puppetmaster.UNAME) as c:
        c.connect_kwargs.password = puppetmaster.PSWD
        result = c.run('cd Desktop/VisualSwarm && '
                       'git pull && '
                       'pipenv install -d --skip-lock -e .',
                       hide=True,
                       pty=False)
        print(result.stdout)

def start_swarm():
    logger.info('Puppetmaster started!')
    with Connection(list(puppetmaster.HOSTS.values())[0], user=puppetmaster.UNAME) as c:
        c.connect_kwargs.password = puppetmaster.PSWD
        c.run('cd Desktop/VisualSwarm && '
              f'ENABLE_CLOUD_LOGGING=1 ROBOT_NAME={list(puppetmaster.HOSTS.keys())[0]} '
              'dtach -n /tmp/tmpdtach '
              'pipenv run vswrm-start-vision',
              hide=True,
              pty=False)
        start_result = c.run('ps ax  | grep "dtach -n /tmp/tmpdtach pipenv run vswrm-start-vision"')
        PID = start_result.stdout.split()[0]
        print(f'Started process with PID: {int(PID)}')
        print('going to sleeeeeep...')
        sleep(15)
        print('Waking up and killing process!')
        end_result = c.run(f'kill -INT {int(PID)}')
        print(end_result.stdout)
