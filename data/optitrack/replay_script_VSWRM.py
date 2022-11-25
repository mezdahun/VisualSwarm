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

for EXPERIMENT_NAME in EXPERIMENT_NAMES:
    # if data is freshly created first summarize it into multidimensional array
    csv_path = os.path.join(data_path, f"{EXPERIMENT_NAME}.csv")
    data_tools.optitrackcsv_to_VSWRM(csv_path, skip_already_summed=True)

    # retreiving data
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)

    # replaying experiment
    plotting_tools.plot_replay_run(summary, data)

plt.show()
