from fabric import Connection
from time import sleep

from visualswarm.contrib import puppetmaster

def start_swarm():
    with Connection(puppetmaster.HOSTS[0], user=puppetmaster.UNAME) as c:
        c.connect_kwargs.password = puppetmaster.PSWD
        c.run('cd Desktop/VisualSwarm && dtach -n /tmp/tmpdtach pipenv run vswrm-start-vision', hide=True, pty=False)
        start_result = c.run('ps ax  | grep "dtach -n /tmp/tmpdtach pipenv run vswrm-start-vision"')
        PID = start_result.stdout.split()[0]
        print(f'Started process with PID: {int(PID)}')
        print('going to sleeeeeep...')
        sleep(5)
        print('Waking up and killing process!')
        end_result = c.run(f'kill -INT {int(PID)}')
        print(end_result.stdout)