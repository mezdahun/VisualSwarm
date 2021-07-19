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
    # Otherwise we will use default values defined below or values passed from env variables individually
    behave_params_dict = {}

    GAM_ENV = os.getenv('GAM')
    if GAM_ENV is not None:
        behave_params_dict['GAM'] = float(GAM_ENV)

    V0_ENV = os.getenv('V0')
    if V0_ENV is not None:
        behave_params_dict['V0'] = float(V0_ENV)

    ALP0_ENV = os.getenv('ALP0')
    if ALP0_ENV is not None:
        behave_params_dict['ALP0'] = float(ALP0_ENV)

    ALP1_ENV = os.getenv('ALP1')
    if ALP1_ENV is not None:
        behave_params_dict['ALP1'] = float(ALP1_ENV)

    BET0_ENV = os.getenv('BET0')
    if BET0_ENV is not None:
        behave_params_dict['BET0'] = float(BET0_ENV)


    BET1_ENV = os.getenv('BET1')
    if BET1_ENV is not None:
        behave_params_dict['BET1'] = float(BET1_ENV)


    KAP_ENV = os.getenv('KAP')
    if KAP_ENV is not None:
        behave_params_dict['KAP'] = float(KAP_ENV)


# Velocity Parameters
GAM = behave_params_dict.get('GAM', 0.1)
V0 = behave_params_dict.get('V0', 125)
ALP0 = behave_params_dict.get('ALP0', 125)
ALP1 = behave_params_dict.get('ALP1', 0.00005)
ALP2 = behave_params_dict.get('ALP2', 0)

# Heading Vector Parameters
BET0 = behave_params_dict.get('BET0', 0.25)
BET1 = behave_params_dict.get('BET1', 0.1)
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
