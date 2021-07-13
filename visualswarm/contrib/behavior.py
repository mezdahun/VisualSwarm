"""
@author: mezdahun
@description: Parameters related to flocking algorithm. Can be passed as an external json file by passing the file path
     in the BEHAVE_PARAMS_JSON_PATH variable. If the path is worn an exception is raised. If nothing is passed the here
     defined default values are used.
"""
import json
import os

# Reading params from json file if requested
BEHAVE_PARAMS_JSON_PATH = os.getenv('BEHAVE_PARAMS_JSON_PATH')
if BEHAVE_PARAMS_JSON_PATH is not None:
    if os.path.isfile(BEHAVE_PARAMS_JSON_PATH):
        with open(BEHAVE_PARAMS_JSON_PATH) as f:
            behave_params_dict = json.load(f)
    else:
        raise Exception('The given parameter json file defined in "BEHAVE_PARAMS_JSON_PATH" is not found!')
else:
    # Otherwise we will use default values defined below
    behave_params_dict = {}

# Velocity Parameters
GAM = behave_params_dict.get('GAM', 0.2)
V0 = behave_params_dict.get('V0', 20)
ALP0 = behave_params_dict.get('ALP0', 50000)
ALP1 = behave_params_dict.get('ALP1', 1)
ALP2 = behave_params_dict.get('ALP2', 0)

# Heading Vector Parameters
BET0 = behave_params_dict.get('BET0', 5000)
BET1 = behave_params_dict.get('BET1', 0.3)
BET2 = behave_params_dict.get('BET2', 0)

# Motor scale heuristics Kappa
KAP = behave_params_dict.get('KAP', 1)


def get_params():
    params = {"GAM": GAM,
              "V0": V0,
              "ALP0": ALP0,
              "ALP1": ALP1,
              "ALP2": ALP2,
              "BET0": BET0,
              "BET1": BET1,
              "BET2": BET2,
              "KAP": KAP}
    return params
