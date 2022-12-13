"""
@description: cPOC for processing optitrack data for publication.
"""
"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
from visualswarm.simulation_tools import data_tools, plotting_tools
import os
import matplotlib.pyplot as plt
import numpy as np

data_path = "/home/david/Desktop/database/OptiTrackCSVs/2B1"
EXPERIMENT_NAMES = []
alphas = [20, 80, 320]
betas = [0.1, 1, 8]
i = 1
for alpha in alphas:
    for beta in betas:
        EXPERIMENT_NAMES.append(f"E2B1{i}_a{alpha}_b{int(beta*10)}")
        i += 1
print(EXPERIMENT_NAMES)
WALL_EXPERIMENT_NAME = "../ArenaBorders_02122022"
indices = [en.split("E2B")[1] for en in EXPERIMENT_NAMES]

wall_ord_tw = [100, 1500]
wall_iid_tw = [100, 1500]
mean_ord_after_wall_m = np.zeros((2, len(alphas), len(betas), np.sum(wall_ord_tw)))
mean_iid_after_wall_m = np.zeros((2, len(alphas), len(betas), np.sum(wall_iid_tw)))
num_clus_matrix = np.zeros((len(alphas), len(betas)))
num_clus_matrix_std = np.zeros((len(alphas), len(betas)))
polrats_over_exps = np.zeros((len(alphas), len(betas), 100))
time_above_pol = np.zeros((len(alphas), len(betas)))
time_in_iid_tolerance = np.zeros((len(alphas), len(betas)))
comv_matrix_final = np.zeros((len(alphas), len(betas)))
comv_matrix_final_std = np.zeros((len(alphas), len(betas)))
ord_matrix_final = np.zeros((len(alphas), len(betas)))
ord_matrix_final_std = np.zeros((len(alphas), len(betas)))
pol_matrix_final = np.zeros((len(alphas), len(betas)))
pol_matrix_final_std = np.zeros((len(alphas), len(betas)))
iid_matrix_final = np.zeros((len(alphas), len(betas)))
iid_matrix_final_std = np.zeros((len(alphas), len(betas)))
coll_times = np.zeros((2, len(alphas), len(betas)))

valid_ts_r, min_iids_r, mean_iids_r, mean_pols_r = [], [], [], []

for ei, EXPERIMENT_NAME in enumerate(EXPERIMENT_NAMES):
    a = float(EXPERIMENT_NAME.split("_")[1].split("a")[1])
    b = float(EXPERIMENT_NAME.split("_")[2].split("b")[1])/10
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
                                                                                                              window_after=300,
                                                                                                              window_before=0)

    mean_ord_after_wall_m[0, alphas.index(a), betas.index(b), :], mean_ord_after_wall_m[1, alphas.index(a), betas.index(b),
                                        :] = data_tools.calculate_avg_metric_after_wall_refl(ord[0, :],
                                                                                             wall_reflection_times,
                                                                                             wall_ord_tw)
    mean_iid_after_wall_m[0, alphas.index(a), betas.index(b), :], mean_iid_after_wall_m[1, alphas.index(a), betas.index(b),
                                        :] = data_tools.calculate_avg_metric_after_wall_refl(mean_iid[0, :],
                                                                                             wall_reflection_times,
                                                                                             wall_iid_tw)

    # valid_ts_iid = data_tools.return_validts_iid(mean_iid[0], iid_of_interest=1200,
    #                                              tolerance=100)

    time_w = 30  # in minutes

    # valid_ts_pol = data_tools.return_validts_pol(mean_pol_vals, pol_thr=0.5)
    # valid_ts_pol = valid_ts_pol[valid_ts_pol>num_t-(time_w*60*30)]
    # valid_ts_pol = [t for t in valid_ts_pol if t in valid_ts]
    #
    # valid_ts_iid = valid_ts_iid[valid_ts_iid>num_t-(time_w*60*30)]
    # valid_ts_iid = [t for t in valid_ts_iid if t in valid_ts]
    #
    time_lim = num_t-(time_w*60*30)
    if time_lim < 0:
        time_lim = 0
    print("Time lim: ", time_lim)
    valid_ts_last_chunk = [t for t in valid_ts if t > time_lim]

    cluster_dict = data_tools.subgroup_clustering(summary, pm, iidm, valid_ts=np.array(valid_ts_last_chunk), runi=0)


    # Calculating polarization time ratios
    pol_ratios = []
    mean_ord_vals = ord[0, :]
    for i in range(100):
        # pol_ratio = np.count_nonzero(mean_pol_vals[valid_ts]>i*0.01) / (len(valid_ts))
        pol_ratio = np.count_nonzero(mean_ord_vals[valid_ts_last_chunk] > i * 0.01) / (len(valid_ts_last_chunk))
        pol_ratios.append(pol_ratio)
    polrats_over_exps[alphas.index(a), betas.index(b), :] = np.array(pol_ratios)


    print("All valid timepoints: ", len(valid_ts))
    print(f"Valid timepoints in last {time_w} minutes: ", len(valid_ts_last_chunk))
    valid_ts_r.append(valid_ts)
    mean_iids_r.append(mean_iid_long)
    mean_pols_r.append(mean_pol_vals_long)
    valid_ts = valid_ts_last_chunk[0:-1]
    # time_above_pol[alphas.index(a), betas.index(b)] = len(valid_ts_pol) / len(valid_ts_last_chunk)
    # time_in_iid_tolerance[alphas.index(a), betas.index(b)] = len(valid_ts_iid) / len(valid_ts_last_chunk)
    num_clus_matrix[alphas.index(a), betas.index(b)] = np.array(np.mean(cluster_dict["num_subgroups"]))
    num_clus_matrix_std[alphas.index(a), betas.index(b)] = np.array(np.std(cluster_dict["num_subgroups"]))
    comv_matrix_final[alphas.index(a), betas.index(b)] = np.mean(com_vel[0, valid_ts])
    comv_matrix_final_std[alphas.index(a), betas.index(b)] = np.std(com_vel[0, valid_ts])
    ord_matrix_final[alphas.index(a), betas.index(b)] = np.mean(ord[0, valid_ts])
    ord_matrix_final_std[alphas.index(a), betas.index(b)] = np.std(ord[0, valid_ts])
    pol_matrix_final[alphas.index(a), betas.index(b)] = np.mean(mean_pol_vals_long[valid_ts])
    pol_matrix_final_std[alphas.index(a), betas.index(b)] = np.std(mean_pol_vals_long[valid_ts])
    iid_matrix_final[alphas.index(a), betas.index(b)] = np.mean(mean_iid_long[valid_ts])
    iid_matrix_final_std[alphas.index(a), betas.index(b)] = np.std(mean_iid_long[valid_ts])
    coll_times[0, alphas.index(a), betas.index(b)] = len(wall_reflection_times)
    coll_times[1, alphas.index(a), betas.index(b)] = len(agent_reflection_times)
    print(f"alpha={a}, beta={b}, ord={ord_matrix_final[alphas.index(a), betas.index(b)]}")

summed_mean_pol_vals = []
for i in range(len(valid_ts_r)):
    valid_ts = valid_ts_r[i]
    mean_pol_vals_long = mean_pols_r[i]
    summed_mean_pol_vals.append(np.mean(mean_pol_vals_long[valid_ts]))

# fig, ax = plt.subplots(1, 2)
# plt.axes(ax[0])
# plt.imshow(pol_matrix_final.T, vmin=0, vmax=0.8)
# plt.title("Mean polarization (w/o collisions)")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)
# plt.axes(ax[1])
# plt.imshow(pol_matrix_final_std.T, vmin=0, vmax=0.8)
# plt.title("STD (over t) polarization (w/o collisions)")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)

# fig, ax = plt.subplots(1, 2)
# plt.axes(ax[0])
# plt.imshow(time_in_iid_tolerance.T*100)
# plt.title("Time in good IID ")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)
# plt.axes(ax[1])
# for i in range(len(alphas)):
#     plt.plot(time_in_iid_tolerance[i, :]*100, label=f"$\\alpha_0$={alphas[i]}")
# plt.xticks([i for i in range(len(betas))], betas)
# plt.xlabel("$\\beta_0$")
# plt.ylabel("time ratio [%]")
# plt.legend()
#
# fig, ax = plt.subplots(1, 2)
# plt.axes(ax[0])
# plt.imshow(time_above_pol.T*100)
# plt.title("Time above 0.8 Pol ")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)
# plt.axes(ax[1])
# for i in range(len(alphas)):
#     plt.plot(time_above_pol[i, :]*100, label=f"$\\alpha_0$={alphas[i]}")
# plt.xticks([i for i in range(len(betas))], betas)
# plt.xlabel("$\\beta_0$")
# plt.ylabel("time ratio [%]")
# plt.legend()
fig, ax = plt.subplots(len(alphas), len(betas))
# mean_mean_ord_after_wall_m = np.mean(mean_ord_after_wall_m, axis=2)
# std_mean_ord_after_wall_m = np.std(mean_ord_after_wall_m, axis=2)
plt.suptitle("Typical Order profile after wall reflection")
for a in range(len(alphas)):
    for b in range(len(betas)):
        plt.axes(ax[a, b])
        plt.title(f"$alpa_0$={alphas[a]}, $beta_0$={betas[b]}")
        plt.plot(mean_ord_after_wall_m[0, a, b, :])
        # plt.fill_between([i for i in range(np.sum(wall_ord_tw))], mean_mean_ord_after_wall_m[0, a, :] - std_mean_ord_after_wall_m[0, a, :],
        #                  mean_mean_ord_after_wall_m[0, a, :] + std_mean_ord_after_wall_m[0, a, :], alpha=0.2)
        plt.vlines(wall_ord_tw[0], np.min(mean_ord_after_wall_m[0, a, b, :]),
                   np.max(mean_ord_after_wall_m[0, a, b, :]), colors="red")
        plt.xlabel("dt [ts]")
        plt.xticks([i for i in range(0, np.sum(wall_ord_tw), 100)], [i for i in range(-wall_ord_tw[0], wall_ord_tw[1], 100)])
        plt.ylabel("order [AU]")



fig, ax = plt.subplots(2, 3)
polrats_over_exps_mean = np.mean(polrats_over_exps, axis=1)
polrats_over_exps_std = np.std(polrats_over_exps, axis=1)
for a, alp in enumerate(alphas):
    plt.axes(ax[0, a])
    plt.title(f"alpha={alp}")
    for b, bet in enumerate(betas):
        plt.plot(polrats_over_exps[a, b], label=f"$alpha_0$={alp}, $beta_0$={bet}")
    plt.legend()
for b, bet in enumerate(betas):
    plt.axes(ax[1, b])
    plt.title(f"beta={bet}")
    for a, alp in enumerate(alphas):
        plt.plot(polrats_over_exps[a, b], label=f"$alpha_0$={alp}, $beta_0$={bet}")
    plt.legend()
plt.xlabel("Order thr. [AU]")
plt.xticks([i for i in range(0, 100, 10)], [i*0.1 for i in range(0, 100, 10)])
plt.ylabel("Time ratio spent above thr.")
plt.title("Time spent above order thr.")


fig, ax = plt.subplots(1, 4)
plt.axes(ax[0])
plt.imshow(num_clus_matrix.T)
plt.title("Number of subgroups")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(len(betas))], betas)
plt.axes(ax[1])
plt.imshow(num_clus_matrix_std.T)
plt.title("STD (over t) # subgroups")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(len(betas))], betas)
plt.axes(ax[2])
for i in range(len(alphas)):
    plt.plot(num_clus_matrix[i, :], label=f"$\\alpha_0$={alphas[i]}")
    plt.fill_between([i for i in range(len(alphas))], num_clus_matrix[i, :]-num_clus_matrix_std[i, :],
                      num_clus_matrix[i, :]+num_clus_matrix_std[i, :], alpha=0.2)
plt.xticks([i for i in range(len(betas))], betas)
plt.xlabel("$\\beta_0$")
plt.ylabel("mean # subgroups")
plt.legend()
plt.axes(ax[3])
for i in range(len(betas)):
    plt.plot(num_clus_matrix[:, i], label=f"$\\beta_0$={betas[i]}")
    plt.fill_between([i for i in range(len(betas))], num_clus_matrix[:, i]-num_clus_matrix_std[:, i],
                      num_clus_matrix[:, i]+num_clus_matrix_std[:, i], alpha=0.2)
plt.xticks([i for i in range(len(betas))], betas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("mean # subgroups")
plt.legend()


fig, ax = plt.subplots(1, 3)
plt.axes(ax[0])
plt.imshow(comv_matrix_final.T)
plt.title("COM velocity (w/o collisions)")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(len(betas))], betas)
plt.axes(ax[1])
plt.imshow(comv_matrix_final_std.T)
plt.title("STD (over t) COM velocity (w/o collisions)")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(len(betas))], betas)
plt.axes(ax[2])
for i in range(len(alphas)):
    plt.plot(comv_matrix_final[i, :], label=f"$\\alpha_0$={alphas[i]}")
    plt.fill_between([i for i in range(len(alphas))], comv_matrix_final[i, :]-comv_matrix_final_std[i, :],
                      comv_matrix_final[i, :]+comv_matrix_final_std[i, :], alpha=0.5)
plt.xticks([i for i in range(len(betas))], betas)
plt.xlabel("$\\beta_0$")
plt.ylabel("velocity (mm/s)")
plt.legend()

fig, ax = plt.subplots(1, 4)
plt.axes(ax[0])
plt.imshow(ord_matrix_final.T, vmin=0, vmax=1)
plt.title("Mean order (w/o collisions)")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(len(betas))], betas)
plt.axes(ax[1])
plt.imshow(ord_matrix_final_std.T, vmin=0, vmax=1)
plt.title("STD (over t) order (w/o collisions)")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(len(betas))], betas)
plt.axes(ax[2])
for i in range(len(alphas)):
    plt.plot(ord_matrix_final[i, :], label=f"$\\alpha_0$={alphas[i]}")
    plt.fill_between([i for i in range(len(alphas))], ord_matrix_final[i, :]-ord_matrix_final_std[i, :],
                      ord_matrix_final[i, :]+ord_matrix_final_std[i, :], alpha=0.5)
plt.xticks([i for i in range(len(betas))], betas)
plt.xlabel("$\\beta_0$")
plt.ylabel("order [au]")
plt.legend()
plt.axes(ax[3])
for i in range(len(betas)):
    plt.plot(ord_matrix_final[:, i], label=f"$\\beta_0$={betas[i]}")
    plt.fill_between([i for i in range(len(betas))], ord_matrix_final[:, i]-ord_matrix_final_std[:, i],
                      ord_matrix_final[:, i]+ord_matrix_final_std[:, i], alpha=0.2)
plt.xticks([i for i in range(len(betas))], betas)
plt.xlabel("$\\alpha_0$")
plt.ylabel("order [au]")
plt.legend()
#
#
# fig, ax = plt.subplots(1, 2)
# plt.axes(ax[0])
# plt.imshow(iid_matrix_final.T)
# plt.title("Mean IID (w/o collisions)")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)
# plt.axes(ax[1])
# plt.imshow(iid_matrix_final_std.T)
# plt.title("STD (over t) IID (w/o collisions)")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)
#
# fig, ax = plt.subplots(2, 1)
# plt.axes(ax[0])
# plt.imshow(coll_times[0, :, :].T)
# plt.title("Wall refl. times")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)
# plt.axes(ax[1])
# plt.imshow(coll_times[1, :, :].T)
# plt.title("Agent refl. times")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)

plt.show()