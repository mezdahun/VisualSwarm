"""VSWRM Thymio Controller. Will only run from Webots terminal!
    -don't forget to setup a venv and install all the packages of visualswarm
    -don't forget to set the python command in webots to the python in your venv
"""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Camera

import os
# Set environment variables for configuration here!
os.environ['ENABLE_SIMULATION'] = str(int(True))
os.environ['SHOW_VISION_STREAMS'] = str(int(True))
os.environ['LOG_LEVEL'] = 'DEBUG'

from visualswarm import app_simulation

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

def setup_sensors():
    global robot

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

def setup_motors():
    global robot

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

def setup_leds():
    global robot

    leds = {}
    leds['top'] = robot.getDevice("leds.top")

    return leds

def setup_camera():
    # create and enable the camera on the robot
    camera = Camera("rPi4_Camera_Module_v2.1")
    sampling_freq = 16  #Hz
    sampling_period = int(1/sampling_freq*1000)

    camera.enable(sampling_period)
    return camera

sensors = setup_sensors()

devices = {}
devices['motors'] = setup_motors()
devices['leds'] = setup_leds()
devices['camera'] = setup_camera()

app_simulation.webots_interface(robot, sensors, devices, timestep, with_control=True)