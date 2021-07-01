"""
@author: mezdahun
@description: Parameters related to the puppetmaster submodule controlling a swarm of robots via SSH with fabric
"""
from getpass import getpass

HOSTS = {'Robot1': '192.168.137.174',
         'Robot2': '192.168.137.3'} # '192.168.0.81'}
UNAME = 'pi'
INSTALL_DIR = '/home/pi/VisualSwarm'
