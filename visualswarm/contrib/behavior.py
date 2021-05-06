"""
@author: mezdahun
@description: Parameters related to flocking algorithm.
"""

# Velocity Parameters
GAM = 0.9  # irl 0.2
V0 = 0.4
ALP0 = 0.55  # overall speed scale (limited by possible motor speed) : irl 0.5
ALP1 = 0.010  # ~ 1 / equilibrium distance : irl 0.015
ALP2 = 0

# Heading Vector Parameters
BET0 = 0.75  # overall responsiveness of heading change (turning "speed") : irl 0.5
BET1 = 0.015
BET2 = 0
