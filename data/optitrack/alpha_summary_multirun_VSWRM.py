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
show_change = "alpha"  # or beta
calc_profiles = False  # slow if true
show_scatters = False
if show_change == "alpha":
    alpha_base = "E21"  # when changing alpha
    alphas = [0, 20, 120, 180, 320]
else:
    alpha_base = "E22"  # when changing beta
    alphas = [0.001, 0.1, 1, 6, 14]
num_runs = 5
runs = [0, 1, 2, 3, 4]
alpha_pattern = os.path.join(data_path, f"{alpha_base}*.csv")
EXPERIMENT_NAMES = [pat.split("/")[-1] for pat in list(glob.glob(alpha_pattern))]
EXPERIMENT_NAMES = [pat.split(".")[0] for pat in EXPERIMENT_NAMES]

WALL_EXPERIMENT_NAME = "ArenaBorders"

wall_ord_tw = [200, 1800]
wall_iid_tw = [200, 1800]
hist_res = 15
mean_ord_after_wall_m = np.zeros((2, len(alphas), num_runs, np.sum(wall_ord_tw)))
mean_iid_after_wall_m = np.zeros((2, len(alphas), num_runs, np.sum(wall_iid_tw)))
num_clus_matrix = np.zeros((len(alphas), num_runs))
acc_matrix_final = np.zeros((len(alphas), num_runs))
acc_matrix_final_std = np.zeros_like(acc_matrix_final)
pol_over_wall_dist = np.zeros((len(alphas), num_runs, 400, 2))
polrats_over_exps = np.zeros((len(alphas), num_runs, 100))
polrats_over_exps_hist = np.zeros((len(alphas), num_runs, hist_res))
iidrats_over_exps = np.zeros((len(alphas), num_runs, 60))
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
if show_scatters:
    fig1, ax1 = plt.subplots(len(alphas), 4, sharey=True, sharex=True)
    fig2, ax2 = plt.subplots(len(alphas), 4, sharey=True, sharex=True)
for ei, EXPERIMENT_NAME in enumerate(EXPERIMENT_NAMES):
    print(f"Processing file {EXPERIMENT_NAME}")
    ri_orig = int(EXPERIMENT_NAME.split("r")[1].split(".")[0]) - 1
    a = int(EXPERIMENT_NAME[3]) - 1

    if a < len(alphas) and ri_orig in runs:
        ri = runs.index(ri_orig)
        # if data is freshly created first summarize it into multidimensional array
        csv_path = os.path.join(data_path, f"{EXPERIMENT_NAME}.csv")
        data_tools.optitrackcsv_to_VSWRM(csv_path, skip_already_summed=True)

        # retreiving data
        summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)

        if WALL_EXPERIMENT_NAME is not None:
            csv_path_walls = os.path.join(data_path, f"{WALL_EXPERIMENT_NAME}_r{ri_orig+1}.csv")
            print(f"Using wall file: ", csv_path_walls)
            data_tools.optitrackcsv_to_VSWRM(csv_path_walls, skip_already_summed=True, dropna=False)
            summay_wall, data_wall = data_tools.read_summary_data(data_path, WALL_EXPERIMENT_NAME+f"_r{ri_orig+1}")
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

        # Calculating timepoints where no reflection occured in 150 steps (150/30=5seconds)
        valid_ts, min_iidm_long, mean_iid_long, mean_pol_vals_long = data_tools.return_metrics_where_no_collision(summary, pm, iidm,
                                                                                                                  0,
                                                                                                                  agent_reflection_times,
                                                                                                                  wall_reflection_times,
                                                                                                                  window_after=150,
                                                                                                                  window_before=0,
                                                                                                                  force_recalculate=False)

        cluster_dict = data_tools.subgroup_clustering(summary, pm, iidm, valid_ts=valid_ts, runi=0)
        num_clus_matrix[a, ri] = np.array(np.mean(cluster_dict["num_subgroups"]))

        if calc_profiles:
            mean_ord_after_wall_m[0, a, ri, :], mean_ord_after_wall_m[1, a, ri, :] = data_tools.calculate_avg_metric_after_wall_refl(ord[0, :], wall_reflection_times, wall_ord_tw)
            mean_iid_after_wall_m[0, a, ri, :], mean_iid_after_wall_m[1, a, ri,
                                                :] = data_tools.calculate_avg_metric_after_wall_refl(mean_iid[0, :],
                                                                                                     wall_reflection_times,
                                                                                                     wall_iid_tw)

        print("before ", len(valid_ts))
        # Filtering datapoints where agents are impossibly fast
        valid_ts = data_tools.filter_high_velocity_points(valid_ts, abs_vel, 0, vel_thr=150)
        print("after high vel", len(valid_ts))
        # Filtering datapoints where agents turning impossibly fast
        valid_ts = data_tools.filter_high_turningrate_points(valid_ts, ma_turning_rates, 0, tr_thr=0.02)
        print("after high tr", len(valid_ts))

        time_w = 15  # in minutes
        if isinstance(valid_ts, list):
            valid_ts = np.array(valid_ts)
        valid_ts_last_chunk = valid_ts[valid_ts>num_t-(time_w*30*60)]
        valid_ts = valid_ts_last_chunk[0:-1]
        #
        # # Histogram of I.I.D matrix
        # plt.axes(ax[a, ri])
        # plt.hist(mean_pol_vals[valid_ts], bins=30

        if show_scatters:
            # # # IID-ORD space
            ax1[a, 0].scatter(mean_iid[0, ::30], ord[0, ::30], c="grey", s=0.2, label="all")
            ax1[a, 0].set_title(f"all ${show_change}_0$={alphas[a]}")
            # plt.axes(ax1[a, 1])
            ax1[a, 1].scatter(mean_iid[0, ::30], ord[0, ::30], c="grey", s=0.2, label="all")
            ax1[a, 1].scatter(mean_iid[0, wall_reflection_times], ord[0, wall_reflection_times], c="red", s=0.3, label="wall reflections")
            ax1[a, 1].set_title(f"wall ${show_change}_0$={alphas[a]}")
            # plt.axes(ax1[a, 2])
            ax1[a, 2].scatter(mean_iid[0, ::30], ord[0, ::30], c="grey", s=0.2, label="all")
            ax1[a, 2].scatter(mean_iid[0, agent_reflection_times], ord[0, agent_reflection_times], c="blue", s=0.3, label="agent reflections")
            ax1[a, 2].set_title(f"agent ${show_change}_0$={alphas[a]}")
            # plt.axes(ax1[a, 3])
            ax1[a, 3].scatter(mean_iid[0, [valid_ts[t] for t in range(0, len(valid_ts), 30)]], ord[0, [valid_ts[t] for t in range(0, len(valid_ts), 30)]],
                        c="green", s=0.1, label="after filtering")
            ax1[a, 3].set_title(f"filtered ${show_change}_0$={alphas[a]}")

            # # X-Y space
            plt.axes(ax2[a, 0])
            plt.scatter(data[0, :, 1, ::30], data[0, :, 3, ::30], c="grey", s=0.2, label="all")
            plt.title(f"all ${show_change}_0$={alphas[a]}")
            plt.axes(ax2[a, 1])
            plt.scatter(data[0, :, 1, ::30], data[0, :, 3, ::30], c="grey", s=0.2, label="all")
            plt.scatter(data[0, :, 1, wall_reflection_times], data[0, :, 3, wall_reflection_times], c="red", s=0.3,
                        label="wall reflections")
            plt.title(f"wall ${show_change}_0$={alphas[a]}")
            plt.axes(ax2[a, 2])
            plt.scatter(data[0, :, 1, ::30], data[0, :, 3, ::30], c="grey", s=0.2, label="all")
            plt.scatter(data[0, :, 1, agent_reflection_times], data[0, :, 3, agent_reflection_times], c="blue", s=0.3,
                        label="agent reflections")
            plt.title(f"agent ${show_change}_0$={alphas[a]}")
            plt.axes(ax2[a, 3])
            plt.scatter(data[0, :, 1, [valid_ts[t] for t in range(0, len(valid_ts), 30)]],
                        data[0, :, 3, [valid_ts[t] for t in range(0, len(valid_ts), 30)]],
                        c="green", s=0.1, label="after filtering")
            plt.title(f"filtered ${show_change}_0$={alphas[a]}")

        # Calculating acceleration values
        print(abs_vel.shape)
        acc = np.diff(abs_vel[0, : , :], axis=-1)
        abs_acc = np.abs(acc)

        # Calculating polarization time ratios
        pol_ratios = []
        mean_ord_vals = ord[0, :]
        for i in range(100):
            # pol_ratio = np.count_nonzero(mean_pol_vals[valid_ts]>i*0.01) / (len(valid_ts))
            pol_ratio = np.count_nonzero(mean_ord_vals[valid_ts] > i * 0.01) / (len(valid_ts))
            pol_ratios.append(pol_ratio)
        polrats_over_exps[a, ri, :] = np.array(pol_ratios)

        # Calculating polarization time ratios histogram
        pol_ratios_h = []
        mean_ord_vals = ord[0, :]
        for i in range(hist_res-1):
            # pol_ratio = np.count_nonzero(mean_pol_vals[valid_ts]>i*0.01) / (len(valid_ts))
            pol_ratio = np.count_nonzero( np.logical_and(i * 1/hist_res < mean_ord_vals[valid_ts], mean_ord_vals[valid_ts] < (i+1)*(1/hist_res))) / (len(valid_ts))
            pol_ratios_h.append(pol_ratio)
        polrats_over_exps_hist[a, ri, :-1] = np.array(pol_ratios_h)

        # Calculating iid time ratios
        iid_ratios = []
        mean_iid_vals = min_iidm[0, :]
        for i in range(0, 600, 10):
            # pol_ratio = np.count_nonzero(mean_pol_vals[valid_ts]>i*0.01) / (len(valid_ts))
            iid_ratio = np.count_nonzero(mean_iid_vals[valid_ts] < i ) / (len(valid_ts))
            iid_ratios.append(iid_ratio)
        iidrats_over_exps[a, ri, :] = np.array(iid_ratios)


        # valid_ts_iid = data_tools.return_validts_iid(mean_iid[0], iid_of_interest=1400,
        #                                              tolerance=25)
        # #

        # valid_ts_pol = data_tools.return_validts_pol(mean_pol_vals, pol_thr=0.5)
        # valid_ts_pol = valid_ts_pol[valid_ts_pol>num_t-(time_w*60*60)]
        # valid_ts_pol = [t for t in valid_ts_pol if t in valid_ts]
        #
        # valid_ts_iid = valid_ts_iid[valid_ts_iid>num_t-(time_w*60*60)]
        # valid_ts_iid = [t for t in valid_ts_iid if t in valid_ts]

        print("All valid timepoints: ", len(valid_ts))
        print(f"Valid timepoints in last {time_w} minutes: ", len(valid_ts_last_chunk))
        valid_ts_r.append(valid_ts)
        mean_iids_r.append(mean_iid_long)
        mean_pols_r.append(mean_pol_vals_long)
        valid_ts_matrix_final[a, ri] = len(valid_ts)

        #time_above_pol[a, ri] = len(valid_ts_pol) / len(valid_ts_last_chunk)
        #time_in_iid_tolerance[a, ri] = len(valid_ts_iid) / len(valid_ts_last_chunk)

        for i in range(399):
            bin_len = 5
            temp_pols = mean_pol_vals[np.logical_and(min_wall_dist >= i*bin_len, min_wall_dist < (i+1)*bin_len)]
            mean_pol_over_wd = np.mean(temp_pols)
            std_pol_over_wd = np.std(temp_pols)
            pol_over_wall_dist[a, ri, i, 0] = mean_pol_over_wd
            pol_over_wall_dist[a, ri, i, 1] = std_pol_over_wd

        acc_matrix_final[a, ri] = np.mean(abs_acc)
        acc_matrix_final_std[a, ri] = np.std(abs_acc)
        comv_matrix_final[a, ri] = np.mean(com_vel[0, valid_ts])
        comv_matrix_final_std[a, ri] = np.std(com_vel[0, valid_ts])
        ord_matrix_final[a, ri] = np.mean(ord[0, valid_ts])
        ord_matrix_final_std[a, ri] = np.std(ord[0, valid_ts])
        pol_matrix_final[a, ri] = np.mean(mean_pol_vals_long[valid_ts])
        pol_matrix_final_std[a, ri] = np.std(mean_pol_vals_long[valid_ts])
        iid_matrix_final[a, ri] = np.mean(mean_iid_long[valid_ts])
        iid_matrix_final_std[a, ri] = np.std(mean_iid_long[valid_ts])
        coll_times[0, a, ri] = len(wall_reflection_times)/num_t
        coll_times[1, a, ri] = len(agent_reflection_times)/num_t
        abs_vel_m_final[a, ri] = abs_vel[..., valid_ts].mean(axis=-1)[0].mean()
        abs_vel_m_final_std[a, ri] = abs_vel[..., valid_ts].mean(axis=-1)[0].std()
        turn_rate_m_final[a, ri] = turning_rates[..., valid_ts].mean(axis=-1)[0].mean()
        turn_rate_final_std[a, ri] = turning_rates[..., valid_ts].mean(axis=-1)[0].std()
        print(f"alpha={alphas[a]}, ri={ri}, ord={ord_matrix_final[a, ri]}")

summed_mean_pol_vals = []
for i in range(len(valid_ts_r)):
    valid_ts = valid_ts_r[i]
    mean_pol_vals_long = mean_pols_r[i]
    summed_mean_pol_vals.append(np.mean(mean_pol_vals_long[valid_ts]))

## mean_time_in_iid_tolerance = time_in_iid_tolerance.mean(axis=1)
## std_time_in_iid_tolerance = time_in_iid_tolerance.std(axis=1)
## fig, ax = plt.subplots(1, 2)
## plt.axes(ax[0])
## plt.imshow(time_in_iid_tolerance.T*100)
## plt.title("Time in good IID ")
## plt.xticks([i for i in range(len(alphas))], alphas)
## plt.yticks([i for i in range(ri)])
## plt.axes(ax[1])
## plt.plot(mean_time_in_iid_tolerance)
## plt.fill_between([i for i in range(len(alphas))], mean_time_in_iid_tolerance-std_time_in_iid_tolerance,
##                   mean_time_in_iid_tolerance+std_time_in_iid_tolerance, alpha=0.5)
## plt.xticks([i for i in range(len(alphas))], alphas)
## plt.xlabel("$\\alpha_0$")
## plt.ylabel("time ratio [%]")
## plt.legend()
##
##
## mean_time_above_pol = time_above_pol.mean(axis=1)
## std_time_above_pol= time_above_pol.std(axis=1)
## fig, ax = plt.subplots(1, 2)
## plt.axes(ax[0])
## plt.imshow(time_above_pol.T*100)
## plt.title("Time above pol. thr. ")
## plt.xticks([i for i in range(len(alphas))], alphas)
## plt.yticks([i for i in range(ri)])
## plt.axes(ax[1])
## plt.plot(mean_time_above_pol)
## plt.fill_between([i for i in range(len(alphas))], mean_time_above_pol-std_time_above_pol,
##                   mean_time_above_pol+std_time_above_pol, alpha=0.5)
## plt.xticks([i for i in range(len(alphas))], alphas)
## plt.xlabel(f"${show_change}_0$")
## plt.ylabel("time ratio [%]")
## plt.legend()

# plt.figure()
# for ai, alpha in enumerate(alphas):
#     plt.plot(pol_over_wall_dist[ai, :, :, 0].mean(axis=0))

fig, ax = plt.subplots(1, len(alphas), sharey=True)
mean_mean_ord_after_wall_m = np.mean(mean_ord_after_wall_m, axis=2)
std_mean_ord_after_wall_m = np.std(mean_ord_after_wall_m, axis=2)
plt.suptitle("Typical Order profile after wall reflection")
for a in range(len(alphas)):
    plt.axes(ax[a])
    plt.title(f"${show_change}_0$={alphas[a]}, r={ri}")
    plt.plot(mean_mean_ord_after_wall_m[0, a, :])
    plt.fill_between([i for i in range(np.sum(wall_ord_tw))], mean_mean_ord_after_wall_m[0, a, :] - std_mean_ord_after_wall_m[0, a, :],
                     mean_mean_ord_after_wall_m[0, a, :] + std_mean_ord_after_wall_m[0, a, :], alpha=0.2)
    plt.vlines(wall_ord_tw[0], np.min(mean_mean_ord_after_wall_m[0, a, :] - std_mean_ord_after_wall_m[0, a, :]),
               np.max(mean_mean_ord_after_wall_m[0, a, :] + std_mean_ord_after_wall_m[0, a, :]), colors="red")
    plt.xlabel("dt [ts]")
    plt.xticks([i for i in range(0, np.sum(wall_ord_tw), 100)], [i for i in range(-wall_ord_tw[0], wall_ord_tw[1], 100)])
    plt.ylabel("order [AU]")

fig, ax = plt.subplots(1, len(alphas), sharey=True)
mean_mean_iid_after_wall_m = np.mean(mean_iid_after_wall_m, axis=2)
std_mean_iid_after_wall_m = np.std(mean_iid_after_wall_m, axis=2)
plt.suptitle("Typical I.I.D. profile after wall reflection")
for a in range(len(alphas)):
    plt.axes(ax[a])
    plt.title(f"${show_change}_0$={alphas[a]}, r={ri}")
    plt.plot(mean_mean_iid_after_wall_m[0, a, :])
    plt.fill_between([i for i in range(np.sum(wall_iid_tw))],
                     mean_mean_iid_after_wall_m[0, a, :] - std_mean_iid_after_wall_m[0, a, :],
                     mean_mean_iid_after_wall_m[0, a, :] + std_mean_iid_after_wall_m[0, a, :], alpha=0.2)
    plt.vlines(wall_iid_tw[0], np.min(mean_mean_iid_after_wall_m[0, a, :] - std_mean_iid_after_wall_m[0, a, :]),
               np.max(mean_mean_iid_after_wall_m[0, a, :] + std_mean_iid_after_wall_m[0, a, :]), colors="red")
    plt.xlabel("dt [ts]")
    plt.xticks([i for i in range(0, np.sum(wall_iid_tw), 100)],
               [i for i in range(-wall_iid_tw[0], wall_iid_tw[1], 100)])
    plt.ylabel("I.I.D [mm]")


# plt.figure()
# plt.imshow(num_clus_matrix.T)

mean_clus = num_clus_matrix.mean(axis=1)
std_clus = num_clus_matrix.std(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(num_clus_matrix.T)
plt.title("Mean number of subgroups")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.xlabel(f"${show_change}_0$")
plt.ylabel(f"runs")
plt.axes(ax[1])
plt.plot(mean_clus)
plt.fill_between([i for i in range(len(alphas))], mean_clus-std_clus,
                  mean_clus+std_clus, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("#")
plt.legend()



mean_ord = ord_matrix_final.mean(axis=1)
std_ord = ord_matrix_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(ord_matrix_final.T)
plt.title("Mean order ")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(ri)])
plt.xlabel(f"${show_change}_0$")
plt.ylabel(f"runs")
plt.axes(ax[1])
plt.plot(mean_ord)
plt.fill_between([i for i in range(len(alphas))], mean_ord-std_ord,
                  mean_ord+std_ord, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("order [AU]")
plt.legend()

mean_iid = iid_matrix_final.mean(axis=1)
std_iid = iid_matrix_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(iid_matrix_final.T)
plt.title("Mean IID")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(num_runs)])
plt.ylabel("runs")
plt.xlabel(f"${show_change}_0$")
plt.axes(ax[1])
plt.plot(mean_iid)
plt.fill_between([i for i in range(len(alphas))], mean_iid-std_iid,
                  mean_iid+std_iid, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("mean IID [mm]")
plt.legend()

mean_av = abs_vel_m_final.mean(axis=1)
std_av = abs_vel_m_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(abs_vel_m_final.T)
plt.title("Mean (over agents and time) absolute velocity \n std over agents")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(num_runs)])
plt.ylabel("runs")
plt.xlabel(f"${show_change}_0$")
plt.axes(ax[1])
plt.plot(mean_av)
plt.fill_between([i for i in range(len(alphas))], mean_av-std_av,
                  mean_av+std_av, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("velocity [mm/ts]")
plt.legend()

mean_acc = acc_matrix_final.mean(axis=1)
std_acc = acc_matrix_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(acc_matrix_final.T)
plt.title("Mean (over agents and time) absolute acceleration \n std over agents")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(num_runs)])
plt.ylabel("runs")
plt.xlabel(f"${show_change}_0$")
plt.axes(ax[1])
plt.plot(std_acc)
plt.fill_between([i for i in range(len(alphas))], std_acc-std_acc,
                  std_acc+std_acc, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("velocity [mm/ts]")
plt.legend()

mean_com = comv_matrix_final.mean(axis=1)
std_com = comv_matrix_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(comv_matrix_final.T)
plt.title("Mean COM velocity ")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(num_runs)])
plt.ylabel("runs")
plt.xlabel(f"${show_change}_0$")
plt.axes(ax[1])
plt.plot(mean_com)
plt.fill_between([i for i in range(len(alphas))], mean_com-std_com,
                  mean_com+std_com, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("mean COM vel [mm/ts]")
plt.legend()

mean_aac = coll_times[1].mean(axis=1)
std_aac = coll_times[1].std(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(coll_times[1].T)
plt.title("Agent-agent collision times")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(num_runs)])
plt.ylabel("runs")
plt.xlabel(f"${show_change}_0$")
plt.axes(ax[1])
plt.plot(mean_aac)
plt.fill_between([i for i in range(len(alphas))], mean_aac-std_aac,
                  mean_aac+std_aac, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("#")


mean_tr = turn_rate_m_final.mean(axis=1)
std_tr = turn_rate_final_std.mean(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(turn_rate_m_final.T)
plt.title("Mean (over agents and time) turning rates")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(num_runs)])
plt.ylabel("runs")
plt.xlabel(f"${show_change}_0$")
plt.axes(ax[1])
plt.plot(mean_tr)
plt.fill_between([i for i in range(len(alphas))], mean_tr-std_tr,
                  mean_tr+std_tr, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(f"${show_change}_0$")
plt.ylabel("Turning rate [mm/ts]")
plt.legend()

plt.figure()
polrats_over_exps_mean = np.mean(polrats_over_exps, axis=1)
polrats_over_exps_std = np.std(polrats_over_exps, axis=1)
for a, alp in enumerate(alphas):
    plt.plot(polrats_over_exps_mean[a, :], label=f"${show_change}_0$={alp}")
    plt.fill_between([i for i in range(0, 100)], polrats_over_exps_mean[a, :]-polrats_over_exps_std[a, :],
                      polrats_over_exps_mean[a, :]+polrats_over_exps_std[a, :], alpha=0.2)
plt.xlabel("Order thr. [AU]")
plt.xticks([i for i in range(0, 100, 10)], [i*0.1 for i in range(0, 100, 10)])
plt.ylabel("Time ratio spent above thr.")
plt.title("Time spent above order thr.")
plt.legend()

fig, ax = plt.subplots(1, len(alphas), sharey=True)
polrats_over_exps_mean = np.mean(polrats_over_exps_hist, axis=1)
polrats_over_exps_std = np.std(polrats_over_exps_hist, axis=1)
for a, alp in enumerate(alphas):
    plt.axes(ax[a])
    if a == 0:
        plt.ylabel("Time ratio spent at order level")
    plt.plot(polrats_over_exps_mean[a, :], label=f"${show_change}_0$={alp}")
    plt.fill_between([i for i in range(0, hist_res)], polrats_over_exps_mean[a, :]-polrats_over_exps_std[a, :],
                      polrats_over_exps_mean[a, :]+polrats_over_exps_std[a, :], alpha=0.2)
    plt.xlabel("Order thr. [AU]")
    plt.title(f"${show_change}_0$={alp}")
    plt.xticks([i for i in range(0, hist_res, 10)], [i*(1/hist_res) for i in range(0, hist_res, 10)])

plt.suptitle("Histogram / time spent at order")
plt.legend()

plt.figure()
iidrats_over_exps_mean = np.mean(iidrats_over_exps, axis=1)
iidrats_over_exps_std = np.std(iidrats_over_exps, axis=1)
for a, alp in enumerate(alphas):
    plt.plot(iidrats_over_exps_mean[a, :], label=f"${show_change}_0$={alp}")
    plt.fill_between([i for i in range(0, 60)], iidrats_over_exps_mean[a, :]-iidrats_over_exps_std[a, :],
                      iidrats_over_exps_mean[a, :]+iidrats_over_exps_std[a, :], alpha=0.2)
plt.xlabel("IID thr. [mm]")
# plt.xticks([i for i in range(0, 3000, 250)])
plt.ylabel("Time ratio spent below thr.")
plt.title("Time spent below iid thr.")
plt.legend()


plt.show()
input()