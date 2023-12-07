import os

# Exploration by rotating towards last visible cue
# Rotation speed
WITH_EXPLORE_ROT = bool(int(os.getenv('WITH_EXPLORE_ROT', '0')))
EXPLORE_ROT_SPEED = int(float(os.getenv('EXPLORE_ROT_SPEED', '50')))
# Continous rotation keeping the velocity but increasing dphi
EXPLORE_ROT_SPEED_CONT = int(float(os.getenv('EXPLORE_ROT_SPEED_CONT', '150')))