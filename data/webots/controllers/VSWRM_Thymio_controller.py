"""VSWRM Thymio Controller. Will only run from Webots terminal!
    -don't forget to setup a venv and install all the packages of visualswarm
    -don't forget to set the python command in webots to the python in your venv
"""

# You may need to import some classes of the controller module. Ex:
from controller import Camera, GPS, Compass, Supervisor

import os

# Set environment variables for configuration here!
os.environ['ENABLE_SIMULATION'] = str(int(True))
os.environ['SHOW_VISION_STREAMS'] = str(int(False))

# logging and performance measure
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['WEBOTS_LOG_PERFORMANCE'] = str(int(False))

# either using multithreading (True) or multiprocessing (False)
os.environ['SPARE_RESCOURCES'] = str(int(True))

# saving simulation data
EXPERIMENT_NAME = 'ExampleExperiment'
os.environ['WEBOTS_SAVE_SIMULATION_DATA'] = str(int(False))
os.environ[
    'WEBOTS_SIM_SAVE_FOLDER'] = f'path/to/folder/{EXPERIMENT_NAME}'

# simulation time limit if 0 then no limit is defined, if larger the simulation will be paused after this time
os.environ['PAUSE_SIMULATION_AFTER'] = '0'
# passing algorithm parameters using json file
os.environ[
    'BEHAVE_PARAMS_JSON_PATH'] = 'path/to/folder/example_params.json'

# visualization
USE_ROBOT_PEN = False


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
    sampling_freq = 60  # Hz
    sampling_period = int(1 / sampling_freq * 1000)
    print(sampling_period)
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

    # example to robot-specific configuration
    if robot.getName() == "Bob":
        os.environ['EXP_MOVEMENT'] = "NoExploration"
    else:
        os.environ['EXP_MOVEMENT'] = "Rotation"

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # setup actuators and sensors
    devices = setup_devices(robot)

    from visualswarm import app_simulation
    app_simulation.webots_entrypoint(robot, devices, timestep, with_control=True)


if __name__ == "__main__":
    main()
