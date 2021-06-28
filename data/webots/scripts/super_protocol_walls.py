"""
Example metaprotocol of changing configuration programatically and calling webots
@description: Script to run a webots world file multiple times with different initial conditions and/or parameters.
"""
import os
import numpy as np

from pprint import pprint
from visualswarm.simulation_tools import webots_tools

KAPPA = 200
# alphas = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
# bethas = [0.1, 0.25, 0.5, 0.75, 1, 1.2]
alphas = [1]  # [0.25, 0.5, 0.75]
bethas = [2.1]

base_path = "C:\\Users\\David\\Documents\\VisualSwarm\\controllers\\blank_controller"
wbt_path = "C:\\Users\\David\\Documents\\VisualSwarm\\worlds" \
           "\\VSWRM_REPRODUCE_FINAL_NOWALLS_4bots_improvedvision_introducewalls.wbt "
num_robots = 4

robot_names = [f"robot{i}" for i in range(num_robots)]

behave_params_path = os.path.join(base_path, 'VAR_behavior_params.json')
initial_condition_path = os.path.join(base_path, 'VAR_initial_conditions.json')
env_config_path = os.path.join(base_path, 'VAR_environment_config.json')

for alpha_0 in alphas:
    for betha_0 in bethas:
        EXPERIMENT_NAME = f"WALLS_An{alpha_0}_Bn{betha_0}_{num_robots}bots"
        print(f"Simulating runs for experiment {EXPERIMENT_NAME} with alpha={alpha_0}, betha={betha_0}")

        # making base link via environmental variables between webots and this script
        # env_config_path should be the same in this script and the controller code in webots
        env_config_dict = {
            'ENABLE_SIMULATION': str(int(True)),
            'SHOW_VISION_STREAMS': str(int(False)),
            'LOG_LEVEL': 'INFO',
            'WEBOTS_LOG_PERFORMANCE': str(int(False)),
            'SPARE_RESCOURCES': str(int(False)),
            'BORDER_CONDITIONS': "Reality",
            'WEBOTS_SAVE_SIMULATION_DATA': str(int(True)),
            'WEBOTS_SAVE_SIMULATION_VIDEO': str(int(True)),  # save video automatically
            'WEBOTS_SIM_SAVE_FOLDER': os.path.join(base_path, 'simulation_data', EXPERIMENT_NAME),
            'PAUSE_SIMULATION_AFTER': '600',
            'PAUSE_BEHAVIOR': 'Quit',  # important to quit when batch simulation scripts are used
            'BEHAVE_PARAMS_JSON_PATH': behave_params_path,
            'INITIAL_CONDITION_PATH': initial_condition_path,
            'USE_ROBOT_PEN': str(int(True))  # enable or disable robot pens
        }
        webots_tools.write_config(env_config_dict, env_config_path)
        print(f"\nGenerated environment config as:\n")
        pprint(env_config_dict)

        # Generate behavior parameters under behave_params path
        behavior_params = {
            "GAM": 0.1,
            "V0": KAPPA,
            "ALP0": KAPPA * alpha_0,
            "ALP1": 0.0006,
            "ALP2": 0,
            "BET0": betha_0,
            "BET1": 0.0006,
            "BET2": 0,
            "KAP": 1
        }
        webots_tools.write_config(behavior_params, behave_params_path)
        print(f"\nGenerated behavior params as:\n")
        pprint(behavior_params)

        num_runs = 1
        print(f"Number of Runs: {num_runs}")

        for run_i in range(num_runs):

            # Change Initial conditions under initial_condition_path
            position_type = {'type': 'uniform',
                             'lim': 2,
                             'center': [0, 0]}
            orientation_type = {'type': 'uniform',
                                'lim': 0.5,  # strong initial polarization to avoid waiting time
                                'center': np.random.uniform(-0.35, 0.35)}
            webots_tools.generate_robot_config(robot_names, position_type, orientation_type, initial_condition_path)

            # call webots to run world file
            print("\n\n ---------- NEW WEBOTS RUN ----------")
            os.system(f"webots --mode=realtime --stdout --stderr {wbt_path}")
            print('Done simulation\n\n\n')

print('Superprotocol Done!')
