"""
@author: mezdahun
@description: Parameters related to simulations in WeBots environment
"""

import os

# Simulation switch, set true to use package from WeBots as a controller package
# Pass it as an environmental variable (in in venv, use a custom activate.bat with additional env vars in it)
ENABLE_SIMULATION = os.getenv('ENABLE_SIMULATION', False)