import os
import string
import random

# You may need to import some classes of the controller module. Ex:
from controller import Camera, GPS, Compass, Supervisor, Pen
from visualswarm.simulation_tools import webots_tools

sbasepath = "/mnt/DATA/mezey/VSWRMSimulationData"  # os.path.dirname(os.path.abspath(__file__))
confbasebath = os.getenv('WEBOTS_CONFBASEPATH', os.path.dirname(os.path.abspath(__file__)))

####### START CONFIG ######

### with external json files (for example for batch scripts)
env_config_path = os.path.join(confbasebath, 'VAR_environment_config.json')
env_config_dict = webots_tools.read_run_config(env_config_path)

### manually

# EXPERIMENT_NAME = 'EQDISTANCE_DV'  # change it before recording data or programatically with input config files
#
# env_config_dict = {
#     'ENABLE_SIMULATION': str(int(True)),  # should be always true if using simulation instead of real robots
#     'SHOW_VISION_STREAMS': str(int(True)),  # visualize what robots see in external openCV window
#     'LOG_LEVEL': 'WARNING',  # verbosity of logging
#     'WEBOTS_LOG_PERFORMANCE': str(int(False)),  # if true, measured times between functional steps will be logged
#     'SPARE_RESCOURCES': str(int(True)),  # Threading for true, multiprocessing for false
#     'BORDER_CONDITIONS': "Reality",  # or "Reality"
#     'WEBOTS_SAVE_SIMULATION_DATA': str(int(True)),
#     'WEBOTS_SAVE_SIMULATION_VIDEO': str(int(True)), # save video automatically
#     'WEBOTS_SIM_SAVE_FOLDER': os.path.join(sbasepath, f'{EXPERIMENT_NAME}'),
#     'PAUSE_SIMULATION_AFTER': '900',  # in seconds
#     'PAUSE_BEHAVIOR': 'Pause',  # or 'Quit'
#     'BEHAVE_PARAMS_JSON_PATH': os.path.join(confbasebath, 'VAR_behavior_params.json'),
#     'INITIAL_CONDITION_PATH': os.path.join(confbasebath, 'VAR_initial_conditions.json'),
#     'USE_ROBOT_PEN': str(int(True)),  # enable or disable robot pens
#     'ROBOT_FOV': '3.8',
#     'EXP_MOVEMENT': 'NoExploration',
#     'WITH_LEADER': str(int(False))
# }

######## END CONFIG #######

webots_tools.initialize_run_w_config(env_config_dict)

INITIAL_CONDITION_PATH = os.getenv('INITIAL_CONDITION_PATH')
WEBOTS_SIM_SAVE_FOLDER = os.getenv('WEBOTS_SIM_SAVE_FOLDER')
EXPERIMENT_NAME = os.path.basename(WEBOTS_SIM_SAVE_FOLDER)
VIDEO_HASH = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
VIDEO_PATH = os.path.join(WEBOTS_SIM_SAVE_FOLDER, f'{EXPERIMENT_NAME}_{VIDEO_HASH}.mp4')
SAVE_VIDEO = bool(int(os.getenv('WEBOTS_SAVE_SIMULATION_VIDEO')))

# visualization
USE_ROBOT_PEN = bool(int(os.getenv('USE_ROBOT_PEN', '0')))

def setup_sensors(robot):
    # Creating sensor structure
    sensors = {}

    # Setting up proximity sensors
    sensors['prox'] = {}
    sensors['prox']['horizontal'] = []

    PROX_UPDATE_FREQ = 10
    for i in range(7):
        device = robot.getDevice(f'prox.horizontal.{i}')
        device.enable(PROX_UPDATE_FREQ)
        sensors['prox']['horizontal'].append(device)

    return sensors


def setup_motors(robot):
    # Creating motor structure
    motors = {}

    motor_left = robot.getDevice("motor.left")
    motor_right = robot.getDevice("motor.right")
    motor_left.setPosition(float('+inf'))
    motor_right.setPosition(float('+inf'))
    motor_left.setVelocity(0)
    motor_right.setVelocity(0)

    motors['left'] = motor_left
    motors['right'] = motor_right

    return motors


def setup_leds(robot):
    leds = {}
    leds['top'] = robot.getDevice("leds.top")

    return leds


def setup_camera(robot):
    # create and enable the camera on the robot
    camera = Camera("rPi4_Camera_Module_v2.1")
    sampling_freq = 8  # Hz
    sampling_period = int(1 / sampling_freq * 1000)
    camera.enable(sampling_period)

    return camera


def setup_monitors(robot):
    # Enable and configure monitoring devices on the robot
    timestep = int(robot.getBasicTimeStep())

    gps = GPS('gps_sensor')
    gps.enable(timestep)

    monitor = {'gps': gps}

    orientation = Compass('orientation_sensor')
    orientation.enable(timestep)

    monitor['orientation'] = orientation
    return monitor


def setup_devices(robot):
    devices = {}
    devices['params'] = {}
    devices['sensors'] = setup_sensors(robot)
    devices['motors'] = setup_motors(robot)
    devices['leds'] = setup_leds(robot)
    devices['camera'] = setup_camera(robot)
    devices['params']['c_height'] = devices['camera'].getHeight()
    devices['params']['c_width'] = devices['camera'].getWidth()
    devices['monitor'] = setup_monitors(robot)
    enable_pen(robot, USE_ROBOT_PEN)
    return devices

def enable_pen(robot, use_pen):
    pen = robot.getDevice("pen")
    pen.write(use_pen)

def main():
    # create the Robot instance.
    robot = Supervisor()
    
    ## example to robot-specific configuration
    if robot.getName() == "robot0": 
        if SAVE_VIDEO:
            robot.movieStartRecording(VIDEO_PATH, 800, 600, 1337, 100, 4, False)
        if bool(int(os.getenv('WITH_LEADER', '1'))):
            os.environ['EXP_MOVEMENT'] = 'NoExploration'
        
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # setup actuators and sensors
    devices = setup_devices(robot)
    
    from visualswarm import app_simulation
    from visualswarm.simulation_tools import webots_tools
    
    webots_tools.initialize_robot_w_config(robot, INITIAL_CONDITION_PATH)
    
    app_simulation.webots_entrypoint(robot, devices, timestep, with_control=True)
    

if __name__ == "__main__":
    main()