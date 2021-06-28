"""
Metaprotocol of changing initial conditions programatically and calling webots to save simulation data
@description: Script to run a webots world file multiple times with different initial conditions and/or parameters.
"""
import os
from visualswarm.simulation_tools import webots_tools

orientation_lims = [1, 2]  # can be any other parameter changed across experiments
experiment_names = ["EXP1", "EXP2"]

base_path = "path-to-controller-folder"
wbt_path = "absolute-path-of-world-file-to-run"
num_robots = 6 # should be matched with world file

robot_names = [f"robot{i}" for i in range(num_robots)]
behave_params_path = os.path.join(base_path, 'VAR_behavior_params.json')
initial_condition_path = os.path.join(base_path, 'VAR_initial_conditions.json')
env_config_path = os.path.join(base_path, 'VAR_environment_config.json')

for i in range(len(experiment_names)):
    EXPERIMENT_NAME = experiment_names[i]
    print(f"Simulating runs for experiment {EXPERIMENT_NAME} with or_lim: {orientation_lims[i]}")

    # making base link via environmental variables between webots and this script
    # env_config_path should be the same in this script and the controller code in webots
    env_config_dict = {
        'ENABLE_SIMULATION': str(int(True)),
        'SHOW_VISION_STREAMS': str(int(False)),
        'LOG_LEVEL': 'DEBUG',
        'WEBOTS_LOG_PERFORMANCE': str(int(False)),
        'SPARE_RESCOURCES': str(int(True)),
        'BORDER_CONDITIONS': "Infinite",
        'WEBOTS_SAVE_SIMULATION_DATA': str(int(True)),
        'WEBOTS_SIM_SAVE_FOLDER': os.path.join(base_path, 'simulation_data', EXPERIMENT_NAME),
        'PAUSE_SIMULATION_AFTER': '450',
        'PAUSE_BEHAVIOR': 'Quit',  # important to quit when batch simulation scripts are used
        'BEHAVE_PARAMS_JSON_PATH': behave_params_path,
        'INITIAL_CONDITION_PATH': initial_condition_path,
        'USE_ROBOT_PEN': str(int(False))  # enable or disable robot pens
    }
    webots_tools.write_config(env_config_dict, env_config_path)

    for run_i in range(1):
        # Generate behavior parameters under behave_params path
        behavior_params = {
            "GAM": 0.1,
            "V0": 150,
            "ALP0": 350,
            "ALP1": 0.00035,
            "ALP2": 0,
            "BET0": 0.6,
            "BET1": 0.00035,
            "BET2": 0,
            "KAP": 1
        }
        webots_tools.write_config(behavior_params, behave_params_path)

        # Change Initial conditions under initial_condition_path
        position_type = {'type': 'uniform',
                         'lim': 5,
                         'center': [0, 0]}
        orientation_type = {'type': 'uniform',
                            'lim': orientation_lims[i],
                            'center': 0.5}
        webots_tools.generate_robot_config(robot_names, position_type, orientation_type, initial_condition_path)

        # call webots to run world file
        print("\n\n ---------- NEW WEBOTS RUN ----------")
        os.system(f"webots --mode=realtime --stdout --stderr {wbt_path}")
        print('Done simulation')

print('Superprotocol Done!')
