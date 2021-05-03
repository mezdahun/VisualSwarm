"""VSWRM Thymio Controller. Will only run from Webots terminal!
    -don't forget to setup a venv and install all the packages of visualswarm
    -don't forget to set the python command in webots to the python in your venv
"""

# You may need to import some classes of the controller module. Ex:
from controller import Robot

import os
# Set environment variables for configuration here!
os.environ['ENABLE_SIMULATION'] = 'True'
os.environ['SHOW_VISION_STREAMS'] = 'False'
os.environ['LOG_LEVEL'] = 'INFO'

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

sensors = setup_sensors()
motors = setup_motors()

app_simulation.webots_interface(robot, sensors, motors, timestep, with_control=True)