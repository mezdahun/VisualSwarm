"""
@author: mezdahun
@description: Parameters related to the puppetmaster submodule controlling a swarm of robots via SSH with fabric
"""
from getpass import getpass

HOSTS = {'Robot1': '192.168.0.81'}
UNAME = 'pi'

# DO NOT STORE PASSWORDS HERE! USE SSH KEYS OR --prompt-for-login-password INSTEAD
PSWD = getpass('Puppetmaster password: ')
