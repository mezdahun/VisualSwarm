"""VSWRM Thymio Controller. Will only run from Webots terminal!
    -don't forget to setup a venv and install all the packages of visualswarm
    -don't forget to set the python command in webots to the python in your venv
"""

# You may need to import some classes of the controller module. Ex:
from controller import Robot, Camera

import os
# Set environment variables for configuration here!
os.environ['ENABLE_SIMULATION'] = str(int(True))
os.environ['SHOW_VISION_STREAMS'] = str(int(False))
os.environ['LOG_LEVEL'] = 'DEBUG'
# either using multithreading or multiprocessing
os.environ['SPARE_RESCOURCES'] = str(int(False))

from visualswarm import app_simulation

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
    sampling_freq = 16  #Hz
    sampling_period = int(1/sampling_freq*1000)
    print(sampling_period)
    camera.enable(sampling_period)

    return camera

def main():

    # create the Robot instance.
    robot = Robot()

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    sensors = setup_sensors(robot)

    devices = {}
    devices['motors'] = setup_motors(robot)
    devices['leds'] = setup_leds(robot)
    devices['camera'] = setup_camera(robot)

    app_simulation.webots_interface(robot, sensors, devices, timestep, with_control=True)
    # while robot.step(timestep) != -1:
        # devices['motors']['left'].setVelocity(9)
        # devices['motors']['right'].setVelocity(0)

if __name__ == "__main__":
    main()