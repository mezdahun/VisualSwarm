"""
@description: cPOC for processing optitrack data for publication.
"""
"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
from visualswarm.simulation_tools import data_tools, plotting_tools
import os
import matplotlib.pyplot as plt

data_path = "/mnt/DATA/mezey/Seafile/SwarmRobotics/VisualSwarm Simulation Data/TESTAFTERMERGE_Exploration_10bots"
EXPERIMENT_NAMES = ["211_repetition3_tracking_data"]
WALL_EXPERIMENT_NAME = "ArenaBorders"

for EXPERIMENT_NAME in EXPERIMENT_NAMES:
    # if data is freshly created first summarize it into multidimensional array
    csv_path = os.path.join(data_path, f"{EXPERIMENT_NAME}.csv")
    data_tools.optitrackcsv_to_VSWRM(csv_path, skip_already_summed=True)

    # retreiving data
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)

    if WALL_EXPERIMENT_NAME is not None:
        csv_path_walls = os.path.join(data_path, f"{WALL_EXPERIMENT_NAME}.csv")
        data_tools.optitrackcsv_to_VSWRM(csv_path, skip_already_summed=True, dropna=False)

        summay_wall, data_wall = data_tools.read_summary_data(data_path, WALL_EXPERIMENT_NAME)

        print(data_wall.shape)
        wall_coordinates = data_wall[0, 0, [1, 3], :]

    else:

        wall_coordinates = None

    # replaying experiment
    plotting_tools.plot_replay_run(summary, data, history_length=1000, wall_coordinates=wall_coordinates)

