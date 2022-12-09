"""
@description: cPOC for processing optitrack data for publication.
"""
"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
from visualswarm.simulation_tools import data_tools, plotting_tools
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

data_path = "/home/david/Desktop/database/OptiTrackCSVs/E2"
EXPERIMENT_NAMES = []
alpha_base = "E21"  # when changing alpha
alphas = [0, 20, 120, 180, 320]
beta_base = "E22"  # when changing beta
betas = [0.001, 0.1, 1]
num_runs = 3
alpha_pattern = os.path.join(data_path, f"{alpha_base}*.csv")
EXPERIMENT_NAMES = [pat.split("/")[-1] for pat in list(glob.glob(alpha_pattern))]
EXPERIMENT_NAMES = [pat.split(".")[0] for pat in EXPERIMENT_NAMES]

WALL_EXPERIMENT_NAME = "../ArenaBorders"

valid_ts_matrix_final = np.zeros((len(alphas), num_runs))
abs_vel_m_final = np.zeros((len(alphas), num_runs))
abs_vel_m_final_std = np.zeros((len(alphas), num_runs))
turn_rate_m_final = np.zeros((len(alphas), num_runs))
turn_rate_final_std = np.zeros((len(alphas), num_runs))
time_above_pol = np.zeros((len(alphas), num_runs))
time_in_iid_tolerance = np.zeros((len(alphas), num_runs))
comv_matrix_final = np.zeros((len(alphas), num_runs))
comv_matrix_final_std = np.zeros((len(alphas), num_runs))
ord_matrix_final = np.zeros((len(alphas), num_runs))
ord_matrix_final_std = np.zeros((len(alphas), num_runs))
pol_matrix_final = np.zeros((len(alphas), num_runs))
pol_matrix_final_std = np.zeros((len(alphas), num_runs))
iid_matrix_final = np.zeros((len(alphas), num_runs))
iid_matrix_final_std = np.zeros((len(alphas), num_runs))
coll_times = np.zeros((2, len(alphas), num_runs))

valid_ts_r, min_iids_r, mean_iids_r, mean_pols_r = [], [], [], []

# only alphas
for ei, EXPERIMENT_NAME in enumerate(EXPERIMENT_NAMES):
    print(f"Processing file {EXPERIMENT_NAME}")
    ri = int(EXPERIMENT_NAME.split("r")[1].split(".")[0]) - 1
    a = int(EXPERIMENT_NAME[3]) - 1

    if a < len(alphas):
        # if data is freshly created first summarize it into multidimensional array
        csv_path = os.path.join(data_path, f"{EXPERIMENT_NAME}.csv")
        data_tools.optitrackcsv_to_VSWRM(csv_path, skip_already_summed=True)

        # retreiving data
        summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)

        if WALL_EXPERIMENT_NAME is not None:
            csv_path_walls = os.path.join(data_path, f"{WALL_EXPERIMENT_NAME}.csv")
            data_tools.optitrackcsv_to_VSWRM(csv_path_walls, skip_already_summed=True, dropna=False)
            summay_wall, data_wall = data_tools.read_summary_data(data_path, WALL_EXPERIMENT_NAME)
            wall_data_tuple = (summay_wall, data_wall)
        else:
            wall_data_tuple = None

        num_t = data.shape[-1]
        print(num_t)

        # replaying experiment
        # plotting_tools.plot_replay_run(summary, data,
        #                                history_length=1000,
        #                                wall_data_tuple=wall_data_tuple,
        #                                step_by_step=False,
        #                                t_start=11000,
        #                                t_step=30,
        #                                use_clastering=False,
        #                                mov_avg_w=60,
        #                                vis_window=3000,
        #                                force_recalculate=False)

        # plotting_tools.plot_summary_over_run(summary, data,
        #                                    history_length=1000,
        #                                    wall_data_tuple=wall_data_tuple,
        #                                    mov_avg_w=2000,
        #                                    force_recalculate=False)

        com_vel, m_com_vel, abs_vel, m_abs_vel, turning_rates, ma_turning_rates, iidm, min_iidm, mean_iid, pm, ord, mean_pol_vals, \
            mean_wall_dist, min_wall_dist, wall_refl_dict, ag_refl_dict, wall_reflection_times, \
            agent_reflection_times, wall_distances, wall_coords_closest \
            = data_tools.return_summary_data(summary, data,
                                             wall_data_tuple=wall_data_tuple, runi=0,
                                             mov_avg_w=30, force_recalculate=False)

        valid_ts, min_iidm_long, mean_iid_long, mean_pol_vals_long = data_tools.return_metrics_where_no_collision(summary, pm, iidm,
                                                                                                                  0,
                                                                                                                  agent_reflection_times,
                                                                                                                  wall_reflection_times,
                                                                                                                  window_after=150,
                                                                                                                  window_before=0)
        print("before ", len(valid_ts))
        valid_ts = data_tools.filter_high_velocity_points(valid_ts, abs_vel, 0, vel_thr=150)
        print("after high vel", len(valid_ts))
        valid_ts = data_tools.filter_high_turningrate_points(valid_ts, ma_turning_rates, 0, tr_thr=0.02)
        print("after high tr", len(valid_ts))

        valid_ts_iid = data_tools.return_validts_iid(mean_iid[0], iid_of_interest=1400,
                                                     tolerance=25)

        time_w = 15  # in minutes
        valid_ts_pol = data_tools.return_validts_pol(mean_pol_vals, pol_thr=0.5)
        valid_ts_pol = valid_ts_pol[valid_ts_pol>num_t-(time_w*60*60)]
        valid_ts_pol = [t for t in valid_ts_pol if t in valid_ts]

        valid_ts_iid = valid_ts_iid[valid_ts_iid>num_t-(time_w*60*60)]
        valid_ts_iid = [t for t in valid_ts_iid if t in valid_ts]

        if isinstance(valid_ts, list):
            valid_ts = np.array(valid_ts)
        valid_ts_last_chunk = valid_ts[valid_ts>num_t-(time_w*60*60)]


        print("All valid timepoints: ", len(valid_ts))
        print(f"Valid timepoints in last {time_w} minutes: ", len(valid_ts_last_chunk))
        valid_ts_r.append(valid_ts)
        mean_iids_r.append(mean_iid_long)
        mean_pols_r.append(mean_pol_vals_long)
        valid_ts = valid_ts[0:-1]
        valid_ts_matrix_final[a, ri] = len(valid_ts)
        time_above_pol[a, ri] = len(valid_ts_pol) / len(valid_ts_last_chunk)
        time_in_iid_tolerance[a, ri] = len(valid_ts_iid) / len(valid_ts_last_chunk)
        comv_matrix_final[a, ri] = np.mean(com_vel[0, valid_ts])
        comv_matrix_final_std[a, ri] = np.std(com_vel[0, valid_ts])
        ord_matrix_final[a, ri] = np.mean(ord[0, valid_ts])
        ord_matrix_final_std[a, ri] = np.std(ord[0, valid_ts])
        pol_matrix_final[a, ri] = np.mean(mean_pol_vals_long[valid_ts])
        pol_matrix_final_std[a, ri] = np.std(mean_pol_vals_long[valid_ts])
        iid_matrix_final[a, ri] = np.mean(mean_iid_long[valid_ts])
        iid_matrix_final_std[a, ri] = np.std(mean_iid_long[valid_ts])
        coll_times[0, a, ri] = len(wall_reflection_times)
        coll_times[1, a, ri] = len(agent_reflection_times)
        abs_vel_m_final[a, ri] = abs_vel.mean(axis=-1)[0].mean()
        abs_vel_m_final_std[a, ri] = abs_vel.mean(axis=-1)[0].std()
        turn_rate_m_final[a, ri] = turning_rates.mean(axis=-1)[0].mean()
        turn_rate_final_std[a, ri] = turning_rates.mean(axis=-1)[0].std()
        print(f"alpha={alphas[a]}, ri={ri}, ord={ord_matrix_final[a, ri]}")

summed_mean_pol_vals = []
for i in range(len(valid_ts_r)):
    valid_ts = valid_ts_r[i]
    mean_pol_vals_long = mean_pols_r[i]
    summed_mean_pol_vals.append(np.mean(mean_pol_vals_long[valid_ts]))

mean_time_in_iid_tolerance = time_in_iid_tolerance.mean(axis=1)
std_time_in_iid_tolerance = time_in_iid_tolerance.std(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(time_in_iid_tolerance.T*100)
plt.title("Time in good IID ")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_time_in_iid_tolerance)
plt.fill_between([i for i in range(len(alphas))], mean_time_in_iid_tolerance-std_time_in_iid_tolerance,
                  mean_time_in_iid_tolerance+std_time_in_iid_tolerance, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("time ratio [%]")
plt.legend()


mean_time_above_pol = time_above_pol.mean(axis=1)
std_time_above_pol= time_above_pol.std(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(time_above_pol.T*100)
plt.title("Time above pol. thr. ")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_time_above_pol)
plt.fill_between([i for i in range(len(alphas))], mean_time_above_pol-std_time_above_pol,
                  mean_time_above_pol+std_time_above_pol, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("time ratio [%]")
plt.legend()


mean_ord = ord_matrix_final.mean(axis=1)
std_ord = ord_matrix_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(ord_matrix_final.T)
plt.title("Mean order ")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_ord)
plt.fill_between([i for i in range(len(alphas))], mean_ord-std_ord,
                  mean_ord+std_ord, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("order [AU]")
plt.legend()

mean_iid = iid_matrix_final.mean(axis=1)
std_iid = iid_matrix_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(iid_matrix_final.T)
plt.title("Mean IID ")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_iid)
plt.fill_between([i for i in range(len(alphas))], mean_iid-std_iid,
                  mean_iid+std_iid, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("mean IID [mm]")
plt.legend()

mean_av = abs_vel_m_final.mean(axis=1)
std_av = abs_vel_m_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(abs_vel_m_final.T)
plt.title("Mean (over agents and time) absolute velocity \n std over agents")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_av)
plt.fill_between([i for i in range(len(alphas))], mean_av-std_av,
                  mean_av+std_av, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("velocity [mm/ts]")
plt.legend()

mean_com = comv_matrix_final.mean(axis=1)
std_com = comv_matrix_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(iid_matrix_final.T)
plt.title("Mean COM velocity ")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_com)
plt.fill_between([i for i in range(len(alphas))], mean_com-std_com,
                  mean_com+std_com, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("mean COM vel [mm/ts]")
plt.legend()

mean_aac = coll_times[1].mean(axis=1)
std_aac = coll_times[1].std(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(iid_matrix_final.T)
plt.title("Agent-agent collision times")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_aac)
plt.fill_between([i for i in range(len(alphas))], mean_aac-std_aac,
                  mean_aac+std_aac, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("#")


mean_tr = turn_rate_m_final.mean(axis=1)
std_tr = turn_rate_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(turn_rate_m_final.T)
plt.title("Mean (over agents and time) turning rates")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.axes(ax[1])
plt.plot(mean_tr)
plt.fill_between([i for i in range(len(alphas))], mean_tr-std_tr,
                  mean_tr+std_tr, alpha=0.5)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("Turning rate [mm/ts]")
plt.legend()

plt.show()