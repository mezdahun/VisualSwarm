"""
Example metaprotocol of changing configuration programatically and calling webots
@description: Script to run a webots world file multiple times with different initial conditions and/or parameters.
"""
import os
import numpy as np

from pprint import pprint
from visualswarm.simulation_tools import webots_tools

# Defining fixed parameters
# Motor scaling
KAPPA = 80
# Simulation timesteps
SIMULATION_TIME = 15 * 60
# Number of robots
num_robots = 10
# Number of repetitions per parameter combo
num_runs = 1

# Name of experiment (batch)
BATCH_NAME = f"{num_robots}bots_testafterlongpause"

# Setting up tuned parameters
alphas = [0.75]
bethas = [6]
FOV_percent = 0.55
FOV = FOV_percent * 2 * np.pi

# Define path for saving
save_path = "/mnt/DATA/mezey/Seafile/SwarmRobotics/VisualSwarm Simulation Data"
save_path = os.path.join(save_path, BATCH_NAME)

# Path of world file
wbt_path = f"/home/mezey/Webots_VSWRM/VisualSwarm/data/webots/VSWRM_WeBots_Project/worlds/VSWRM_{num_robots}Bots_OptiTrackArena.wbt"

# Controller folder path
cont_folder = "/home/mezey/Webots_VSWRM/VisualSwarm/data/webots/VSWRM_WeBots_Project/controllers/VSWRM-controller"

robot_names = [f"robot{i}" for i in range(num_robots)]

donei = 1
for alpi, alpha_0 in enumerate(alphas):
    for beti, betha_0 in enumerate(bethas):
        for run_i in range(num_runs):

            # Define paths for configuration files
            # Initial conditions, behavioral parameters and simulation parameters will be saved in json files
            base_path = f"{cont_folder}/conf_{alpi}{beti}{run_i}{np.random.randint(9, 100, 1)}"
            if not os.path.isdir(base_path):
                os.makedirs(base_path, exist_ok=True)
            behave_params_path = os.path.join(base_path, 'VAR_behavior_params.json')
            initial_condition_path = os.path.join(base_path, 'VAR_initial_conditions.json')
            env_config_path = os.path.join(base_path, 'VAR_environment_config.json')

            # Change Initial conditions under initial_condition_path
            position_type = {'type': 'uniform',
                             'lim': 2.5,
                             'center': [-2.5, -1.25]}
            orientation_type = {'type': 'uniform',
                                'lim': 0.25,  # strong initial polarization to avoid waiting time
                                'center': np.pi/2}
            webots_tools.generate_robot_config(robot_names, position_type, orientation_type, initial_condition_path)
            print("Generated robot config for all runs!")

            EXPERIMENT_NAME = f"TestAfterLongPause_An{alpha_0}_Bn{betha_0}_{num_robots}bots_FOV{FOV}"
            print(f"Simulating runs for experiment {EXPERIMENT_NAME} with alpha0={alpha_0}, betha0={betha_0}")

            # Generate behavior parameters under behave_params path
            behavior_params = {
                "GAM": 0.1,
                "V0": KAPPA,
                "ALP0": KAPPA * alpha_0,
                "ALP1": 0.0012,
                "ALP2": 0,
                "BET0": betha_0,
                "BET1": 0.0012,
                "BET2": 0,
                "KAP": 1
            }
            webots_tools.write_config(behavior_params, behave_params_path)
            print(f"\nGenerated behavior params as:\n")
            pprint(behavior_params)

            print(f"Number of Runs: {num_runs}")

            # making base link via environmental variables between webots and this script
            # env_config_path should be the same in this script and the controller code in webots
            env_config_dict = {
                'ENABLE_SIMULATION': str(int(True)),
                'SHOW_VISION_STREAMS': str(int(False)),
                'LOG_LEVEL': 'ERROR',
                'WEBOTS_LOG_PERFORMANCE': str(int(False)),
                'SPARE_RESCOURCES': str(int(True)),
                'BORDER_CONDITIONS': "Reality",
                'WEBOTS_SAVE_SIMULATION_DATA': str(int(True)),
                'WEBOTS_SAVE_SIMULATION_VIDEO': str(int(True)),  # save video automatically
                'WEBOTS_SIM_SAVE_FOLDER': os.path.join(save_path, EXPERIMENT_NAME),
                'PAUSE_SIMULATION_AFTER': str(SIMULATION_TIME),
                'PAUSE_BEHAVIOR': 'Quit',  # important to quit when batch simulation scripts are used
                'BEHAVE_PARAMS_JSON_PATH': behave_params_path,
                'INITIAL_CONDITION_PATH': initial_condition_path,
                'USE_ROBOT_PEN': str(int(True)),  # enable or disable robot pens
                'ROBOT_FOV': str(FOV),
                'EXP_MOVEMENT': 'NoExploration',  # RandomWalk, NoExploration
                'WITH_LEADER': str(int(False))
            }

            webots_tools.write_config(env_config_dict, env_config_path)
            print(f"\nGenerated environment config as:\n")
            pprint(env_config_dict)

            # call webots to run world file
            print("\n\n ---------- NEW WEBOTS RUN ----------")
            os.system(f"WEBOTS_CONFBASEPATH={base_path} nohup env WEBOTS_CONFBASEPATH={base_path} webots --mode=realtime --stdout --stderr --minimize {wbt_path} &")
            print('Started simulation\n\n\n')
            print(f'PROGRESS: {(donei/(len(alphas)*len(bethas)*num_runs))*100}%')
            donei += 1

print('Superprotocol Done!')
