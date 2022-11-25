"""
@author: mezdahun
@description: Parameters related to the puppetmaster submodule controlling a swarm of robots via SSH with fabric

example command on robot: ENABLE_CLOUD_LOGGING=0 ENABLE_CLOUD_STORAGE=0 SAVE_VISION_VIDEO=1 SHOW_VISION_STREAMS=1
FLIP_CAMERA=0 ROBOT_FOV=3.8 LOG_LEVEL=DEBUG SAVE_CNN_TRAINING_DATA=0 ROBOT_NAME=Robot2 vswrm-start-vision
"""

# Select only a few robots, uncomment
# selected_robots = [3, 8]

# select all robots uncomment this line
selected_robots = [i+1 for i in range(10)]

ALL_HOSTS = {'Robot1': '192.168.0.194',
             'Robot2': '192.168.0.176',
             'Robot3': '192.168.0.173',
             'Robot4': '192.168.0.110',
             'Robot5': '192.168.0.177',
             'Robot6': '192.168.0.122',
             'Robot7': '192.168.0.153',
             'Robot8': '192.168.0.169',
             'Robot9': '192.168.0.145',
             'Robot10': '192.168.0.103'}

HOSTS = {}
for rid in selected_robots:
    HOSTS[f"Robot{rid}"] = ALL_HOSTS[f"Robot{rid}"]


WEBCAM_HOSTS = {
    'Birdseye Cam': '192.168.100.105'
}
UNAME = 'pi'
INSTALL_DIR = '/home/pi/VisualSwarm'
