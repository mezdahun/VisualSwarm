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
# Fixed speed during random walk exploration
V_EXP_RW = 12

# Possible absolute angle change in a given timestep during exploration (in radian)
DPSI_MAX_EXP = 0.25

# Random drawing precision
DPSI_DRAW_PRECISION = 0.01
