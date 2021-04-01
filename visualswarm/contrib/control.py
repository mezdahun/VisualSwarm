"""
@author: mezdahun
@description: Motor Control related parameters
"""
# Serial port on which the Thymio is available
THYMIO_DEVICE_PORT = "/dev/ttyACM0"

# Motor scale correction to put the motor scales into the right region
MOTOR_SCALE_CORRECTION = 40

# Distance between 2 contact points of wheel and ground in m
B = 0.11

# Maximum motor speed of thymio2
MAX_MOTOR_SPEED = 500

# Exploration
# waiting some time before exploration if no input (sec)
WAIT_BEFORE_SWITCH_MOVEMENT = 1

# Random walk time step to change direction (sec)
RW_DT = 1

# Fixed speed during random walk exploration
V_EXP_RW = 6

# Possible absolute angle change in a given timestep during exploration (in radian)
DPSI_MAX_EXP = 1.5
