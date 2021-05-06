"""
@author: mezdahun
@description: Parameters related to simulations in WeBots environment
"""

import os

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
