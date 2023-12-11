import os


##### EXPLORATION #####
# Exploration by stationary rotating towards last visible cue
# Turn ON/OFF exploration by stationary rotating towards last visible cue
WITH_EXPLORE_ROT = bool(int(os.getenv('WITH_EXPLORE_ROT', '0')))
# Rotation speed in motor units
EXPLORE_ROT_SPEED = int(float(os.getenv('EXPLORE_ROT_SPEED', '50')))

# Continous rotation keeping the absolute velocity but with fixed dphi
# keeping only the direction towards dphi in last timestep
# Turn ON/OFF exploration by rotating towards last visible cue
WITH_EXPLORE_ROT_CONT = bool(int(os.getenv('WITH_EXPLORE_ROT_CONT', '0')))
# Rotation absolute speed in motor units
EXPLORE_ROT_SPEED_CONT = int(float(os.getenv('EXPLORE_ROT_SPEED_CONT', '100')))
# Fixed rotation angle in radian (dphi)
EXPLORE_ROT_THETA_CONT = float(os.getenv('EXPLORE_ROT_THETA_CONT', '0.75'))


##### FRONT-BACK OSCILLATIONS #####
# Turn ON/OFF limited backwards movement
WITH_LIMITED_BACKWARDS = bool(int(os.getenv('WITH_LIMITED_BACKWARDS', '0')))
# Maximum absolut backwards speed in motor units
MAX_BACKWARDS_SPEED = int(float(os.getenv('MAX_BACKWARDS_SPEED', '50')))
# Turn ON/OFF mAX Forward speed
WITH_LIMITED_FORWARD = bool(int(os.getenv('WITH_LIMITED_FORWARD', '0')))
# Maximum absolut forward speed in motor units
MAX_FORWARD_SPEED = int(float(os.getenv('MAX_FORWARD_SPEED', '160')))
# Turn on/off sigmid mask function for acceleration response
WITH_SIGMOID_MASK_ACC = bool(int(os.getenv('WITH_SIGMOID_MASK_ACC', '0')))
# Sigmoid steepness for acceleration response mask
SIGMOID_MASK_ACC_STEEP = int(float(os.getenv('SIGMOID_MASK_ACC_STEEP', '5')))


##### LAZY TURNING #####
# Turn ON/OFF stationary turning behavior below velocity threshold
WITH_STAT_TURNING = bool(int(os.getenv('WITH_STAT_TURNING', '0')))
# Velocity threshold below which stationary turning is triggered in motor units
STAT_TURN_VEL_THRES = int(float(os.getenv('STAT_TURN_VEL_THRES', '30')))
# Fixed turning speed in motor units (forward)
STAT_TURN_SPEED = int(float(os.getenv('STAT_TURN_SPEED', '400')))
# Fixed turning speed in motor units (backward)
STAT_TURN_SPEED_BACK = int(float(os.getenv('STAT_TURN_SPEED_BACK', '200')))
# Turn on/off sigmoid mask function for turning response
WITH_SIGMOID_MASK_TURN = bool(int(os.getenv('WITH_SIGMOID_MASK_TURN', '0')))
# Sigmoid steepness for acceleration response mask
SIGMOID_MASK_TURN_STEEP = int(float(os.getenv('SIGMOID_MASK_TURN_STEEP', '5')))