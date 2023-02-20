"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
import os
import matplotlib.pyplot as plt
import numpy as np

from visualswarm.simulation_tools import data_tools, plotting_tools

# if data is freshly created first summarize it into multidimensional array
BASE_PATH = '/media/david/DMezeySCIoI/VSWRMData/Webots'
#BASE_PATH = '/home/david/Desktop/database/velcompare'
BATCH_NAME = "RealExperiments_EXP2.2_10bots_FOV0475_fric06_maxspeed55_NOcamdistort_slip0_fixedvision_tallhelo_rescaledturn"
EXPERIMENT_NAMES = ["Exp22_An0.75_Bn1_10bots"]
WALL_EXPERIMENT_NAME = "ArenaBordersSynth"

for EXPERIMENT_NAME in EXPERIMENT_NAMES:
    data_path = os.path.join(BASE_PATH, BATCH_NAME, EXPERIMENT_NAME)
    data_path_walls = os.path.join(BASE_PATH, BATCH_NAME, WALL_EXPERIMENT_NAME)

    # if data is freshly created first summarize it into multidimensional array
    # retreiving data
    data_tools.summarize_experiment(data_path, EXPERIMENT_NAME, skip_already_summed=True)
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)
    t_len = data.shape[-1]
    #
    summay_wall, data_wall = data_tools.read_summary_data(data_path_walls, WALL_EXPERIMENT_NAME)


    plotting_tools.plot_replay_run(summary, data, t_start=20600,#t_len-10,
                                   # t_end=-1,
                                   t_step=60, vis_window=1200, #t_len-15,
                                   wall_data_tuple=(summay_wall, data_wall),
                                   show_wall_distance=False,
                                   force_recalculate=False,
                                   show_COM_vel=True,
                                   use_clastering=False,
                                   turn_thr=0.011,
                                   step_by_step=False,
                                   meas_ts=1)

plt.show()
