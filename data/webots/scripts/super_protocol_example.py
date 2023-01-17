"""
Example metaprotocol of changing configuration programatically and calling webots
@description: Script to run a webots world file multiple times with different initial conditions and/or parameters.
"""
import os
import subprocess
import time

import numpy as np

from pprint import pprint
from visualswarm.simulation_tools import webots_tools

# Defining fixed parameters
# Motor scaling
KAPPA = 80
# Simulation timesteps
SIMULATION_TIME_MIN = 35  # in minutes
SIMULATION_TIME = SIMULATION_TIME_MIN * 60
# Number of robots
num_robots = 10
# Number of repetitions per parameter combo
num_runs = 4
# Number of maximum parallel webots processes
num_max_processes = 5

# Name of experiment (batch)
BATCH_NAME = f"RealExperiments_EXP2.1_{num_robots}bots_withMass135_DynVisionNoise_MotorNoise_FOV205deg"

# Setting up tuned parameters
# Field of View
FOV_percent = 0.57  # 210deg
FOV = FOV_percent * 2 * np.pi

# Behavioral Parameters (tuning alpha and Beta)
# Exp.2.1 - Tuning Alpha while Beta large
alpha_1 = beta_1 = 0.0014
alphas = [0, 0.25, 1.5, 2.25, 4]
bethas = [8]


# Define path for saving
save_path = "/mnt/DATA/mezey/Seafile/SwarmRobotics/VisualSwarm Simulation Data"
save_path = os.path.join(save_path, BATCH_NAME)

# Path of world file
wbt_path = f"/home/mezey/Webots_VSWRM/VisualSwarm/data/webots/VSWRM_WeBots_Project/worlds/VSWRM_{num_robots}Bots_OptiTrackArena_withPhysics.wbt"

# Controller folder path
cont_folder = "/home/mezey/Webots_VSWRM/VisualSwarm/data/webots/VSWRM_WeBots_Project/controllers/VSWRM-controller"

robot_names = [f"robot{i}" for i in range(num_robots)]

donei = 1
for alpi, alpha_0 in enumerate(alphas):
    for beti, betha_0 in enumerate(bethas):
        run_i = 0
        while run_i < num_runs:

            time.sleep(1)
            num_running_processes = len(subprocess.run(["pgrep", "webots"], stdout=subprocess.PIPE).stdout.decode('utf-8').split("\n")) - 1
            num_running_processes = num_running_processes / 2
            print(f"Found {num_running_processes} webots processes, {num_max_processes-num_running_processes} free slots available.")

            if num_running_processes < num_max_processes:
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

                EXPERIMENT_NAME = f"Exp21_An{alpha_0}_Bn{betha_0}_{num_robots}bots"
                print(f"Simulating runs for experiment {EXPERIMENT_NAME} with alpha0={alpha_0}, betha0={betha_0}")

                # Generate behavior parameters under behave_params path
                behavior_params = {
                    "GAM": 0.1,
                    "V0": KAPPA,
                    "ALP0": KAPPA * alpha_0,
                    "ALP1": alpha_1,
                    "ALP2": 0,
                    "BET0": betha_0,
                    "BET1": beta_1,
                    "BET2": 0,
                    "KAP": 1
                }
                webots_tools.write_config(behavior_params, behave_params_path)
                print(f"\nGenerated behavior params as:\n")
                pprint(behavior_params)

                print(f"Number of Runs: {num_runs}")
                if run_i <= 2:
                    save_video = True
                else:
                    save_video = False

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
                    'WEBOTS_SAVE_SIMULATION_VIDEO': str(int(save_video)),  # save video automatically
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
                time.sleep(1)

                # call webots to run world file
                print("\n\n ---------- NEW WEBOTS RUN ----------")
                # To start N webots processes parallel
                os.system(f"WEBOTS_CONFBASEPATH={base_path} nohup env WEBOTS_CONFBASEPATH={base_path} webots --mode=realtime --batch --stdout --stderr --minimize {wbt_path} &")
                # To start webots processing one by one after each other
                # os.system(f"WEBOTS_CONFBASEPATH={base_path} webots --mode=realtime --stdout --stderr --minimize {wbt_path} &")
                time.sleep(1)
                print('Started simulation\n\n\n')
                print(f'PROGRESS: {(donei/(len(alphas)*len(bethas)*num_runs))*100}%')
                donei += 1
                run_i += 1
            else:
                print("Maximum number of webots processes are already running. waiting for a free slot.")
                print(f'PROGRESS: {(donei / (len(alphas) * len(bethas) * num_runs)) * 100}%')
                time.sleep(60)

print('Superprotocol Done!')
