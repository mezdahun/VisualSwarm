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
MAX_BACKWARDS_SPEED = int(float(os.getenv('MAX_BACKWARDS_SPEED', '35')))
# Turn ON/OFF mAX Forward speed
WITH_LIMITED_FORWARD = bool(int(os.getenv('WITH_LIMITED_FORWARD', '0')))
# Maximum absolut forward speed in motor units
MAX_FORWARD_SPEED = int(float(os.getenv('MAX_FORWARD_SPEED', '250')))


##### SIGMOID MASKS #####
# Turn on/off sigmid mask function for acceleration response
WITH_SIGMOID_MASK_ACC = bool(int(os.getenv('WITH_SIGMOID_MASK_ACC', '0')))
# Sigmoid steepness for acceleration response mask
SIGMOID_MASK_ACC_STEEP = int(float(os.getenv('SIGMOID_MASK_ACC_STEEP', '15')))
# Turn on/off sigmoid mask function for turning response
WITH_SIGMOID_MASK_TURN = bool(int(os.getenv('WITH_SIGMOID_MASK_TURN', '0')))
# Sigmoid steepness for acceleration response mask
SIGMOID_MASK_TURN_STEEP = int(float(os.getenv('SIGMOID_MASK_TURN_STEEP', '15')))


##### LAZY TURNING #####
# Turn ON/OFF stationary turning behavior below velocity threshold
WITH_IMPR_TURNING = bool(int(os.getenv('WITH_IMPR_TURNING', '0')))
# Turning towards overall more excitation when only less than certain amount of blobs is visible (only 1 e.g.)
STAT_TURN_NUM_BLOB_THRES = int(float(os.getenv('STAT_TURN_NUM_BLOB_THRES', '2')))
# Fixed multiplier for turning speed in motor units when moving backwards
STAT_TURN_SPEED_BACK = int(float(os.getenv('STAT_TURN_SPEED_BACK', '200')))
# Centralization Rate, i.e. how quick the robot turns towards the center of mass of the visual blobs if less than
# STAT_TURN_NUM_BLOB_THRES blobs are visible. This is also multiplied by the speed so faster robots turn faster
CENTRALIZE_SPEED = float(os.getenv('CENTRALIZE_SPEED', '0.05'))


##### SELECTIVE BLOB FILTERING #####
# Filtering visual blobs selectively
# (1) only when there are more than 2 visible blobs
# (2) we take the COM of visual blobs on the retina and make removal candidates
#     the last N blobs in the list sorted according to their distance from the COM
# (3) we remove blobs that are below a certain size from the candidates
# Turn ON/OFF selective blob filtering
WITH_SELECTIVE_BLOB_FILTERING = bool(int(os.getenv('WITH_SELECTIVE_BLOB_FILTERING', '0')))
# Remove blobs below this size selectively (only when they are far from other blobs)
MIN_BLOB_SIZE = int(float(os.getenv('MIN_BLOB_SIZE', '8')))
# When ordering accordin to COM distance the last N blob will be candidate for removal
N_BLOB_CANDIDATES = int(float(os.getenv('N_BLOB_CANDIDATES', '2')))