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