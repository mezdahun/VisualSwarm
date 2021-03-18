"""
@author: mezdahun
@description: Parameters related to flocking algorithm.
"""

# Velocity Parameters
GAM = 0.2
V0 = 0
ALP0 = 0.5  # overall responsiveness of velocity change (acceleration "speed")
ALP1 = 0.015  # ~ 1 / equilibrium distance
ALP2 = 0

# Heading Vector Parameters
BET0 = 0.3  # overall responsiveness of heading change (turning "speed")
BET1 = 0.015
BET2 = 0

# Normalizing Velocity
V_MAX_ALG = 6000
V_MAX_PHYS = 400

# Normalizing dpsi
DPSI_MAX_ALG = 700
DPSI_MAX_PHYS = 1
