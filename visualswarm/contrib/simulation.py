"""
@author: mezdahun
@description: Parameters related to simulations in WeBots environment
"""

import os
import numpy as np

# Simulation switch, set true to use package from WeBots as a controller package
# Pass it as an environmental variable (in in venv, use a custom activate.bat with additional env vars in it)
ENABLE_SIMULATION = bool(int(os.getenv('ENABLE_SIMULATION', '0')))

# Update frequency of sensors
UPFREQ_PROX_HORIZONTAL = 10  #in Hz

# Switch to use either multithreading or multiprocessing module during
# webots simulations. Advantage with multiprocessing module is that it
# can be scaled and distributed on e.g. a cluster of machines, although
# this is not yet implemented. If multithreading is used webots auto-
# matically limits the available rescources for the simulation, therefore
# it is safer but slower. Deafult value is safe mode, using threading.
SPARE_RESCOURCES = bool(int(os.getenv('SPARE_RESCOURCES', '1')))

# Max Thymio motor speed in webots environment
MAX_WEBOTS_MOTOR_SPEED = 9.53

# Zero angle direction, robot orientation will be calculated as a difference from this vector
# if an axis is disable in Webots, write nan there here too
WEBOTS_ZERO_ORIENTATION = [1, np.nan, 0]

# Robot forward axis
WEBOTS_ROBOT_FWD_AXIS = np.array([0, 0, 1])

# calculating which coordinate will decide on orientation sign
WEBOTS_ORIENTATION_SIGN_IDX = np.nonzero(WEBOTS_ROBOT_FWD_AXIS[~np.isnan(np.array(WEBOTS_ZERO_ORIENTATION))])[0][0]

# Saving simulation data if true
WEBOTS_SAVE_SIMULATION_DATA = bool(int(os.getenv('WEBOTS_SAVE_SIMULATION_DATA', '0')))

# Saving simulation data to
WEBOTS_SIM_SAVE_FOLDER = os.getenv('WEBOTS_SIM_SAVE_FOLDER')

# Limit simulation end uniformly
PAUSE_SIMULATION_AFTER = int(os.getenv('PAUSE_SIMULATION_AFTER'))

WEBOTS_LOG_PERFORMANCE = bool(int(os.getenv('WEBOTS_LOG_PERFORMANCE', '0')))