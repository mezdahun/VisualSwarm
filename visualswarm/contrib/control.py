"""
@author: mezdahun
@description: Motor Control related parameters
"""
# Serial port on which the Thymio is available
THYMIO_DEVICE_PORT = "/dev/ttyACM0"

# Motor scale correction to put the motor scales into the right region
MOTOR_SCALE_CORRECTION = 75

# Distance between 2 contact points of wheel and ground in m
B = 0.11

# Maximum motor speed of thymio2
MAX_MOTOR_SPEED = 500