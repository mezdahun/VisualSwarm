"""
@author: mezdahun
@description: tools to configure robot parameters and initialize webots environment
"""
import json
import logging
import numpy as np
import os
from itertools import combinations
from visualswarm import env

# setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(env.LOG_LEVEL)


def read_robot_config(robot, config_path):
    """reading robot-specific configuration into a data structure from a json file under config_path according to
    the name of the robot"""
    with open(config_path, "r") as jf:
        robot_config = json.load(jf)

    return robot_config.get(robot.getName())


def read_run_config(env_config_path):
    """reading run configuration holding environmental variable values into a data structure from a json file
    under env_config_path"""
    with open(env_config_path, "r") as jf:
        run_config = json.load(jf)
        return run_config


def write_config(env_config, env_config_path):
    """simple wrapper around json.dump to make code cleaner"""
    with open(env_config_path, 'w') as config_f:
        json.dump(env_config, config_f, indent=4)


def initialize_run_w_config(env_config_dict):
    """setting all env variables according to a configuration divtionary in a batch"""
    logger.info("Configuring run...")
    for var_name, value in env_config_dict.items():
        logger.info(f"Setting env var {var_name} to {value}")
        os.environ[var_name] = value


def generate_robot_config(robot_names, position_type, orientation_type, config_path,
                          max_tries=5000, min_dist=0.5):
    """generating robot specific configuration file according to the passed parameters

        Args:
            robot_names: list of string robot names in the world file
            position_type (dict): including information about the robot positions in a dictionary, example:

                {'type': 'uniform',
                'lim': 5,
                'center': [0, 0]}

            orientation_type (dict) including information about the robot orientation, example:

                {'type': 'uniform',
                'lim': 3.14,
                'center': 0.5}

            config_path: path of json file to generate the configurations in
        Params:
            max_tries: the maximum iterations to try to generate position matrix
            min_dist: minimum robot-to-robot distance to keep in m
        Returns:
            None, generates file
    """

    robot_config = {}
    num_robots = len(robot_names)

    if position_type['type'] == 'uniform':
        i = 0
        while i < max_tries:
            P = np.random.rand(num_robots, 2) * position_type['lim']
            if all(np.linalg.norm(p - q) > min_dist for p, q in combinations(P, 2)):
                P[:, 0] -= ((position_type['lim'] / 2) - position_type['center'][0])
                P[:, 1] -= ((position_type['lim'] / 2) - position_type['center'][1])
                break
            i += 1

    if orientation_type['type'] == 'uniform':
        O = np.random.rand(num_robots, 1) * orientation_type['lim']
        O -= ((orientation_type['lim'] / 2) - orientation_type['center'])

    for i, robot_name in enumerate(robot_names):
        robot_config[robot_name] = {'translation': [P[i, 0], 0, P[i, 1]],
                                    'rotation': [0, 1, 0, O[i, 0]]}

    with open(config_path, 'w') as param_f:
        json.dump(robot_config, param_f, indent=4)


def initialize_robot_w_config(robot, config_path):
    robot_config = read_robot_config(robot, config_path)
    robot_node = robot.getFromDef("VswrmThymhio2")
    translation_field = robot_node.getField('translation')
    translation_field.setSFVec3f(robot_config['translation'])
    rotation_field = robot_node.getField('rotation')
    rotation_field.setSFRotation(robot_config['rotation'])


def teleport_to_center_if_needed(robot, position, threshold=1):
    main_arena = robot.getFromDef("MainArena")
    trf = main_arena.getField("translation")
    floor_size = main_arena.getField("floorSize")
    tr_val = trf.getSFVec3f()
    size_val = floor_size.getSFVec2f()

    if position[1] > tr_val[0] + (size_val[0] / 2) - threshold \
            or position[1] < tr_val[0] - (size_val[0] / 2) + threshold \
            or position[3] > tr_val[2] + (size_val[1] / 2) - threshold \
            or position[3] < tr_val[2] - (size_val[1] / 2) + threshold:
        trf.setSFVec3f([position[1], 0, position[3]])


def write_ER_timestamp(robot_name, config_filepath, run_number, timestamp):
    """Appending emergency timestamps to a json file in the saving directory"""
    basepath = os.path.dirname(config_filepath)
    filepath = os.path.join(basepath, f'{robot_name}_run{run_number}_ERtimes.json')

    if os.path.isfile(filepath):
        with open(filepath, "r") as jf:
            timestamp_dict = json.load(jf)
    else:
        timestamp_dict = {'ERtimes': []}

    timestamp_dict['ERtimes'].append(timestamp)
    with open(filepath, 'w') as jf:
        json.dump(timestamp_dict, jf, indent=4)
