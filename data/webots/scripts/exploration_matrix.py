import os

from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from visualswarm.simulation_tools import data_tools, plotting_tools

BASE_PATH = '/media/david/DMezeySCIoI/VSWRMData/Webots'
BATCH_NAME = "Experiment22_tuningBeta_Webots"
RUN_BASE_NAME = "Exp22"
EXPERIMENT_FOLDER = os.path.join(BASE_PATH, BATCH_NAME)
FOV = "3.455751918948773"
WALL_EXPERIMENT_NAME = "ArenaBordersSynth"
data_path_walls = os.path.join(BASE_PATH, BATCH_NAME, WALL_EXPERIMENT_NAME)
summay_wall, data_wall = data_tools.read_summary_data(data_path_walls, WALL_EXPERIMENT_NAME)
wall_data_tuple = (summay_wall, data_wall)

iid_path = os.path.join(EXPERIMENT_FOLDER, "iid.npy")
pol_path = os.path.join(EXPERIMENT_FOLDER, "pol.npy")

alphas = [0.75]#, 0.5, 0.75, 1, 1.25, 1.5] #, 3, 5]
betas = [0.01, 0.1, 1, 6, 14] #, 0.5, 0.75, 1] #, 3, 5]
num_runs = 5

num_clus_matrix = np.zeros((num_runs, len(alphas), len(betas)))
num_clus_matrix_std = np.zeros((num_runs, len(alphas), len(betas)))
ord_matrix_final = np.zeros((num_runs, len(alphas), len(betas)))
ord_matrix_final_std = np.zeros((num_runs, len(alphas), len(betas)))
pol_matrix_final = np.zeros((num_runs, len(alphas), len(betas)))
pol_matrix_final_std = np.zeros((num_runs, len(alphas), len(betas)))

donei = 0
num_data_points = len(alphas)*len(betas)
for ai, alpha in enumerate(alphas):
    for bi, beta in enumerate(betas):
        for runi in range(num_runs):
            print(f"Experiment with alpha: {alpha} and beta {beta}")
            EXPERIMENT_NAME = f"{RUN_BASE_NAME}_An{alpha}_Bn{beta}_10bots"  #_FOV{FOV}"
            data_path = os.path.join(EXPERIMENT_FOLDER, EXPERIMENT_NAME)
            data_tools.summarize_experiment(data_path, EXPERIMENT_NAME, skip_already_summed=True)
            summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)

            num_t = data.shape[-1]

            com_vel, m_com_vel, abs_vel, m_abs_vel, turning_rates, ma_turning_rates, iidm, min_iidm, mean_iid, pm, ord, mean_pol_vals, \
                mean_wall_dist, min_wall_dist, wall_refl_dict, ag_refl_dict, wall_reflection_times, \
                agent_reflection_times, wall_distances, wall_coords_closest \
                = data_tools.return_summary_data(summary, data,
                                                 wall_data_tuple=wall_data_tuple, runi=runi,
                                                 mov_avg_w=30, force_recalculate=False)

            valid_ts, min_iidm_long, mean_iid_long, mean_pol_vals_long = data_tools.return_metrics_where_no_collision(
                summary, pm, iidm,
                runi,
                agent_reflection_times,
                wall_reflection_times,
                window_after=150,
                window_before=0)

            time_w = 10  # in minutes
            time_lim = num_t - (time_w * 60 * 30)
            if time_lim < 0:
                time_lim = 0
            print("Time lim: ", time_lim)
            valid_ts = [t for t in valid_ts if t > time_lim]

            cluster_dict = data_tools.subgroup_clustering(summary, pm, iidm, valid_ts=np.array(valid_ts),
                                                          runi=runi)

            ord_matrix_final[runi, alphas.index(alpha), betas.index(beta)] = np.mean(ord[runi, valid_ts])
            ord_matrix_final_std[runi, alphas.index(alpha), betas.index(beta)] = np.std(ord[runi, valid_ts])
            pol_matrix_final[runi, alphas.index(alpha), betas.index(beta)] = np.mean(mean_pol_vals_long[valid_ts])
            pol_matrix_final_std[runi, alphas.index(alpha), betas.index(beta)] = np.std(mean_pol_vals_long[valid_ts])
            num_clus_matrix[runi, alphas.index(alpha), betas.index(beta)] = np.array(np.mean(cluster_dict["num_subgroups"]))
            num_clus_matrix_std[runi, alphas.index(alpha), betas.index(beta)] = np.array(np.std(cluster_dict["num_subgroups"]))

            donei += 1
            print(f"Process: {donei/num_data_points*100}%")


#### Polarization
# fig, ax = plt.subplots(1, 2)
# plt.axes(ax[0])
# plt.imshow(np.mean(pol_matrix_final, axis=0).T, vmin=0, vmax=0.8)
# plt.title("Mean polarization (w/o collisions)")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)
# plt.axes(ax[1])
# plt.imshow(np.mean(pol_matrix_final_std, axis=0).T, vmin=0, vmax=0.8)
# plt.title("STD (over t) polarization (w/o collisions)")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(len(betas))], betas)

#### Order Parameter
fig, ax = plt.subplots(1, 4)
plt.axes(ax[0])
ord_matrix_final = np.mean(ord_matrix_final, axis=0)
ord_matrix_final_std = np.mean(ord_matrix_final_std, axis=0)
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

#### Number of Subgroups
num_clus_matrix = np.mean(num_clus_matrix, axis=0)
num_clus_matrix_std = np.mean(num_clus_matrix_std, axis=0)
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


plt.show()