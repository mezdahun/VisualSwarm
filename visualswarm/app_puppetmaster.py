from fabric import Connection
from fabric import ThreadingGroup as Group
from time import sleep

from visualswarm.contrib import puppetmaster
from getpass import getpass
# Imports the Cloud Logging client library

import logging
# # setup logging
# logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

PSWD = getpass('Puppetmaster password: ')

def reinstall_robots():
    """Reinstall dependencies of robots from scratch (when something goes wrong). It takes time."""
    logger.info("Reinstalling robots' virtual environment...")
    swarm = Group(*list(puppetmaster.HOSTS.values()), user=puppetmaster.UNAME)
    for c in swarm:
        c.connect_kwargs.password = PSWD
        result = c.run(f'cd {puppetmaster.INSTALL_DIR} && '
                       'git pull && '
                       'pipenv --rm && '
                       'pipenv install -d --skip-lock -e .',
                       hide=True,
                       pty=False)
        print(result.stdout)

def update_robots():
    """Update dependencies of robots (when new package is used)"""
    logger.info("Updating robots' virtual environment...")
    swarm = Group(*list(puppetmaster.HOSTS.values()), user=puppetmaster.UNAME)
    for c in swarm:
        c.connect_kwargs.password = PSWD
        result = c.run(f'cd {puppetmaster.INSTALL_DIR} && '
                       'git pull && '
                       'pipenv install -d --skip-lock -e .',
                       hide=True,
                       pty=False)
        print(result.stdout)

def vswrm_start(c, robot_name):
    """Start VSWRM app on a single robot/connection"""
    c.connect_kwargs.password = PSWD
    c.run(f'cd {puppetmaster.INSTALL_DIR} && '
          'git pull && '
          f'ENABLE_CLOUD_LOGGING=1 ROBOT_NAME={robot_name} LOG_LEVEL=DEBUG '
          'dtach -n /tmp/tmpdtach '
          'pipenv run vswrm-start-vision',
          hide=True,
          pty=False)

def vswrm_stop(c):
    """Stop VSWRM app on a single robot/connection"""
    c.connect_kwargs.password = PSWD
    start_result = c.run('ps ax  | grep "/bin/vswrm-start-vision"')
    PID = start_result.stdout.split()[0] # get PID of first subrocess of vswrm
    # sending INT SIG to any of the subprocesses will trigger graceful exit (equivalent to KeyboardInterrup)
    c.run(f'kill -INT -{int(PID)}')


def start_swarm():
    """Start VSWRM app on a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    logger.info('Puppetmaster started!')
    swarm = Group(*list(puppetmaster.HOSTS.values()), user=puppetmaster.UNAME)
    print(swarm)
    for connection in swarm:
        robot_name = list(puppetmaster.HOSTS.keys())[list(puppetmaster.HOSTS.values()).index(connection.host)]
        print(f'Start VSWRM on {robot_name} with host {connection.host}')
        vswrm_start(connection, robot_name)

    getpass('VSWRM started on swarm. Press any key to stop the swarm')

    logger.info('Killing VSWRM processes by collected PIDs...')
    for connection in swarm:
        robot_name = list(puppetmaster.HOSTS.keys())[list(puppetmaster.HOSTS.values()).index(connection.host)]
        logger.info(f'Stop VSWRM on {robot_name} with host {connection.host}')
        vswrm_stop(connection)

def shutdown_swarm(shutdown='shutdown'):
    """Shutdown/Reboot a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    swarm = Group(*list(puppetmaster.HOSTS.values()), user=puppetmaster.UNAME)
    for connection in swarm:
        connection.connect_kwargs.password = PSWD
        logger.info(f'Shutdown robot with IP {connection.host}')
        connection.sudo(f'{shutdown} -h now')

def shutdown_robots():
    """ENTRYPOINT Shutdown a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    shutdown_swarm()

def restart_robots():
    """ENTRYPOINT Reboot a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    shutdown_swarm(shutdown='reboot')

    # with Connection(list(puppetmaster.HOSTS.values())[0], user=puppetmaster.UNAME) as c:
    #     c.connect_kwargs.password = puppetmaster.PSWD
    #     c.run('cd Desktop/VisualSwarm && '
    #           'git pull && '
    #           f'ENABLE_CLOUD_LOGGING=1 ROBOT_NAME={list(puppetmaster.HOSTS.keys())[0]} '
    #           'dtach -n /tmp/tmpdtach '
    #           'pipenv run vswrm-start-vision',
    #           hide=True,
    #           pty=False)
    #     start_result = c.run('ps ax  | grep "dtach -n /tmp/tmpdtach pipenv run vswrm-start-vision"')
    #     PID = start_result.stdout.split()[0]
    #     print(f'Started process with PID: {int(PID)}')
    #     print('going to sleeeeeep...')
    #     sleep(15)
    #     print('Waking up and killing process!')
    #     end_result = c.run(f'kill -INT {int(PID)}')
    #     print(end_result.stdout)