"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
import os
import matplotlib.pyplot as plt
from visualswarm.simulation_tools import data_tools, plotting_tools

# if data is freshly created first summarize it into multidimensional array
BASE_PATH = '/mnt/DATA/mezey/Seafile/SwarmRobotics/VisualSwarm Simulation Data'
BATCH_NAME = "TESTAFTERMERGE_Exploration_10bots"
EXPERIMENT_NAMES = ["TestAfterLongPause_An3_Bn3_10bots_FOV3.455751918948773"]

for EXPERIMENT_NAME in EXPERIMENT_NAMES:
    data_path = os.path.join(BASE_PATH, BATCH_NAME, EXPERIMENT_NAME)

    # if data is freshly created first summarize it into multidimensional array
    # retreiving data
    data_tools.summarize_experiment(data_path, EXPERIMENT_NAME, skip_already_summed=True)
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)

    plotting_tools.plot_replay_run(summary, data)

plt.show()
