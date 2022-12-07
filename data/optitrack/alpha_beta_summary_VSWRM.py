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
                                                                                                              window_after=600,
                                                                                                              window_before=0)
    print("Valid timepoints: ", len(valid_ts))
    valid_ts_r.append(valid_ts)
    mean_iids_r.append(mean_iid_long)
    mean_pols_r.append(mean_pol_vals_long)
    valid_ts = valid_ts[0:-1]
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

fig, ax = plt.subplots(1, 3)
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