from fabric import ThreadingGroup as Group

from visualswarm.contrib import puppetmaster
from getpass import getpass

import os
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
        try:
            c.connect_kwargs.password = PSWD
            result = c.run(f'cd {puppetmaster.INSTALL_DIR} && '
                           'git pull && '
                           'pipenv install -d --skip-lock -e .',
                           pty=False)
            print(result.stdout)
        except Excetion as e:
            logger.error(f'Error during updating robot: {c}')
            logger.error(f'Error: {e}')


def vswrm_start(c, robot_name):
    """Start VSWRM app on a single robot/connection"""
    EXP_ID = os.getenv('EXP_ID', 'noexpid')
    c.connect_kwargs.password = PSWD
    c.run(f'cd {puppetmaster.INSTALL_DIR} && '
          'git pull && '
          f'ENABLE_CLOUD_LOGGING=0 ENABLE_CLOUD_STORAGE=1 SAVE_VISION_VIDEO=1 SHOW_VISION_STREAMS=0 '
          f'ROBOT_NAME={robot_name} EXP_ID={EXP_ID} LOG_LEVEL=DEBUG FLIP_CAMERA=0 ROBOT_FOV=3.8 BET0=10 ALP0=180 '
          'dtach -n /tmp/tmpdtach '
          'pipenv run vswrm-start')


def vswrm_stop(c):
    """Stop VSWRM app on a single robot/connection"""
    c.connect_kwargs.password = PSWD
    start_result = c.run('ps -x  | grep "/bin/vswrm-start"')
    PID = start_result.stdout.split()[0]  # get PID of first subrocess of vswrm
    print(PID)
    # sending INT SIG to any of the subprocesses will trigger graceful exit (equivalent to KeyboardInterrup)
    c.run(f'cd {puppetmaster.INSTALL_DIR} && touch release.txt && sleep 2 && kill -INT {int(PID)} && rm -rf release.txt')


def start_swarm():
    """Start VSWRM app on a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    logger.info('Puppetmaster started!')
    swarm = Group(*list(puppetmaster.HOSTS.values()), user=puppetmaster.UNAME)
    print(swarm)
    for connection in swarm:
        robot_name = list(puppetmaster.HOSTS.keys())[list(puppetmaster.HOSTS.values()).index(connection.host)]
        print(f'Start VSWRM on {robot_name} with host {connection.host}')
        try:
            vswrm_start(connection, robot_name)
        except Exception as e:
            logger.error(f'Could not start VSWRM on robot: {connection}')
            logger.error(f'Error: {e}')

    getpass('VSWRM started on swarm. Press any key to stop the swarm')

    logger.info('Killing VSWRM processes by collected PIDs...')
    for connection in swarm:
        robot_name = list(puppetmaster.HOSTS.keys())[list(puppetmaster.HOSTS.values()).index(connection.host)]
        logger.info(f'Stop VSWRM on {robot_name} with host {connection.host}')
        try:
            vswrm_stop(connection)
        except Exception as e:
            logger.error(f'Could not stop VSWRM on robot: {connection}')
            logger.error(f'Error: {e}')


def shutdown_swarm(shutdown='shutdown'):
    """Shutdown/Reboot a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    swarm = Group(*list(puppetmaster.HOSTS.values()), user=puppetmaster.UNAME)
    for connection in swarm:
        try:
            connection.connect_kwargs.password = PSWD
            logger.info(f'Shutdown robot with IP {connection.host}')
            connection.sudo(f'{shutdown} -h now')
        except Exception as e:
            logger.error(f'Could not stop VSWRM on robot: {connection}')
            logger.error(f'Error: {e}')


def shutdown_robots():
    """ENTRYPOINT Shutdown a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    shutdown_swarm()


def restart_robots():
    """ENTRYPOINT Reboot a swarm of robots defined with HOSTS in contrib.puppetmaster"""
    shutdown_swarm(shutdown='reboot')
