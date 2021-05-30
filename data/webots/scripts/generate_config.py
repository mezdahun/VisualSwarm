"""
@description: Script to generate robot-specific configuration file (json) to pass robot positions and
orientations to webots.
"""
from visualswarm.simulation_tools import webots_tools

target_path = 'C:\\Users\\David\\Documents\\VisualSwarm\\controllers\\blank_controller\\robot_config_highpolpolswarm3.json'
num_robots = 6


position_type = {'type': 'uniform',  # distribution of points
                 'lim': 5,           # size of uniform box
                 'center': [0, 0]}   # center of uniform box

orientation_type = {'type': 'uniform',  # distribtion of heading angles
                    'lim': 1,        # possible deviation of heading angles of agents (with each other)
                    'center': 0.5}      # polarization direction of swarm

webots_tools.generate_robot_config([f'robot{i}' for i in range(num_robots)], position_type, orientation_type,
                                   target_path)

