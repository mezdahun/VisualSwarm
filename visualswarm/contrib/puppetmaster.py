"""
@author: mezdahun
@description: Parameters related to the puppetmaster submodule controlling a swarm of robots via SSH with fabric

example command on robot: ENABLE_CLOUD_LOGGING=0 ENABLE_CLOUD_STORAGE=0 SAVE_VISION_VIDEO=1 SHOW_VISION_STREAMS=1
FLIP_CAMERA=0 ROBOT_FOV=3.8 LOG_LEVEL=DEBUG SAVE_CNN_TRAINING_DATA=0 ROBOT_NAME=Robot2 vswrm-start-vision
"""

HOSTS = {'Robot1': '192.168.100.162',
         'Robot2': '192.168.100.179',
         'Robot3': '192.168.100.131',
         'Robot4': '192.168.100.170',
         'Robot5': '192.168.100.181'}
UNAME = 'pi'
INSTALL_DIR = '/home/pi/VisualSwarm'
