"""
@description: cPOC for processing optitrack data for publication.
"""
from visualswarm.simulation_tools import data_tools, plotting_tools
import os

data_path = "/home/david/Desktop/database/Experiment1/Experiment1/B0Fixed"
EXPERIMENT_NAMES = ["EXP1_A0_09_B0_125_r1_w1"]
WALL_EXPERIMENT_NAME = "w1"

for EXPERIMENT_NAME in EXPERIMENT_NAMES:
    # if data is freshly created first summarize it into multidimensional array
    csv_path = os.path.join(data_path, f"{EXPERIMENT_NAME}.csv")
    data_tools.optitrackcsv_to_VSWRM(csv_path, skip_already_summed=True)

    # retreiving data
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)
    t_len = data.shape[-1]

    if WALL_EXPERIMENT_NAME is not None:
        csv_path_walls = os.path.join(data_path, f"{WALL_EXPERIMENT_NAME}.csv")
        data_tools.optitrackcsv_to_VSWRM(csv_path_walls, skip_already_summed=False, dropna=False)
        summay_wall, data_wall = data_tools.read_summary_data(data_path, WALL_EXPERIMENT_NAME)
        data_wall = data_tools.resample_wall_coordinates(summay_wall, data_wall, dxy=5, with_plot=False)
        wall_data_tuple = (summay_wall, data_wall)
    else:
        wall_data_tuple = None

    com_vel, m_com_vel, abs_vel, m_abs_vel, turning_rates, ma_turning_rates, iidm, min_iidm, mean_iid, pm, ord, mean_pol_vals, \
        mean_wall_dist, min_wall_dist, wall_refl_dict, ag_refl_dict, wall_reflection_times, \
        agent_reflection_times, wall_distances, wall_coords_closest \
        = data_tools.return_summary_data(summary, data,
                                         wall_data_tuple=wall_data_tuple, runi=0,
                                         mov_avg_w=30, turn_thr=0.023, wall_vic_thr=175,
                                         force_recalculate=False)

    # # mining rotation Y values from raw summary data
    # import matplotlib.pyplot as plt
    # import numpy as np
    # t_start = 0
    # t_end = 18000
    # agent = 0
    # plt.figure()
    # coll_vec = np.abs(np.diff(data[0, agent, 5, :]))
    # coll_times = np.where(coll_vec > 0.9)[0]
    # wall_distances, wall_coords_closest, _ = data_tools.calculate_distances_from_walls(summary, data, summay_wall, data_wall)
    # coll_times_w = [t for t in coll_times if wall_distances[0, agent, t] < 200]
    # # plotting the coll_vec and putting a red dot where the collision is
    # plt.plot(coll_vec)
    # # putting a red dot at y=1 where the collision is according to coll_times
    # plt.plot(coll_times, np.ones(coll_times.shape), "ro")
    # # putting a blue dot at y=1 where the collision is according to coll_times_w
    # plt.plot(coll_times_w, np.ones(len(coll_times_w)), "bo")
    # plt.show()

    # data_tools.mine_reflection_times_drotY(data, summary, summay_wall, data_wall,
    #                             rotation_threshold=0.19, wall_dist_thr=180, agent_dist_thr=275,
    #                             with_plotting=True)



    # print(turning_rates.shape)
    # print(ma_turning_rates.shape)
    #
    # t_start = 2000
    # t_end = 4000
    # agent = 1
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(turning_rates[0, agent, t_start:t_end])
    # plt.plot(ma_turning_rates[0, agent, t_start:t_end], color="black", linewidth=3)
    # plt.show()


    # valid_ts, min_iidm_long, mean_iid_long, mean_pol_vals_long = data_tools.return_metrics_where_no_collision(
    #     summary, pm, iidm,
    #     0,
    #     agent_reflection_times,
    #     wall_reflection_times,
    #     window_after=600,
    #     window_before=0)

    # replaying experiment
    t_start = 50000
    vis_window_width = 600  # 200 ts
    smoothing_width = 30  # 30 ts
    clustering = False
    interactive = False
    #
    plotting_tools.plot_replay_run(summary, data,
                                   history_length=20,
                                   wall_data_tuple=wall_data_tuple,
                                   step_by_step=interactive,
                                   t_start=vis_window_width + t_start,
                                   # t_end=-1,
                                   t_step=smoothing_width,
                                   use_clastering=clustering,
                                   show_clustering=clustering,
                                   show_wall_distance=True,
                                   mov_avg_w=30,
                                   vis_window=vis_window_width,
                                   force_recalculate=False,
                                   video_save_path=None,  # f"{data_path}/{EXPERIMENT_NAME}",
                                   show_COM_vel=True,
                                   show_polarization=False,
                                   turn_thr=0.023,
                                   meas_ts=1
                                   )
