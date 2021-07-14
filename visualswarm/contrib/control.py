"""
@author: mezdahun
@description: Motor Control related parameters
"""
import os

# MOTOR INTERFACE
# Serial port on which the Thymio is available
THYMIO_DEVICE_PORT = "/dev/ttyACM0"

# Motor scale correction to put the motor scales into the right region
# MOTOR_SCALE_CORRECTION = 25

# Distance between 2 contact points of wheel and ground in m
B = 0.11

# Maximum motor speed of thymio2
MAX_MOTOR_SPEED = 500


# EXPLORATION REGIMES
# Exploration mode, possible values: 'RandomWalk', 'Rotation', 'NoExploration'
EXP_MOVE_TYPE = os.getenv('EXP_MOVEMENT', 'NoExploration')

# waiting some time before exploration if no input (sec)
WAIT_BEFORE_SWITCH_MOVEMENT = 1

# Status LEDs
EXPLORE_STATUS_RGB = (32, 32, 32)
BEHAVE_STATUS_RGB = (0, 0, 0)
EMERGENCY_STATUS_RGB = (32, 0, 0)


# RANDOM WALK
# Random walk time step to change direction (sec)
RW_DT = 3.5

# Fixed speed during random walk exploration
V_EXP_RW = 125

# Possible absolute angle change in a given timestep during RW exploration (in radian)
# if zero the exploration is just moving in a straight line
DPSI_MAX_EXP = 3

# ROTATION
# Motor speed (per motor) during rotation (rotation speed)
ROT_MOTOR_SPEED = 50

# Rotation direction, possible values: 'Left', 'Right', 'Random'
ROT_DIRECTION = 'Left'


# OBSTACLE DETECTION
# If set to "Infinite" the robots will teleport upon boundary condition
BORDER_CONDITION = os.getenv('BORDER_CONDITIONS', 'Reality')
# Avoidance turn direction
AVOID_TURN_DIRECTION = 'Various'  # or 'Various'

# Emergency monitoring in Hz (maximum value is 10Hz, on which Thymio is updating these values)
EMERGENCY_CHECK_FREQ = 10

# Threshold value on horizontal proximity sensors that triggers obstacle avoidance
EMERGENCY_PROX_THRESHOLD = 3000

# angle to turn away from obstacle during obstacle avoidance.
turn_angle_correction = 15
desired_alignemnt_angle = 30
OBSTACLE_TURN_ANGLE = desired_alignemnt_angle + turn_angle_correction

# PENDULUM TRAP
# sensor value threshold below which the left and right sensor values are said to be symmetric
SYMMETRICITY_THRESHOLD = 500
# sensor value below which the middle frontal sensor is ignored when the obstacle is symmetric
UNCONTINOUTY_THRESHOLD = 3000
# angle to turn with to get out of pendulum traps
PENDULUM_TRAP_ANGLE = 90
