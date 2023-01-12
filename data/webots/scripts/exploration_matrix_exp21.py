import os

from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from visualswarm.simulation_tools import data_tools, plotting_tools

BASE_PATH = '/media/david/DMezeySCIoI/VSWRMData/Webots'
BATCH_NAME = "Experiment21_tuningAlpha_Webots"
RUN_BASE_NAME = "Exp21"
EXPERIMENT_FOLDER = os.path.join(BASE_PATH, BATCH_NAME)
FOV = "3.455751918948773"
WALL_EXPERIMENT_NAME = "ArenaBordersSynth"
data_path_walls = os.path.join(BASE_PATH, BATCH_NAME, WALL_EXPERIMENT_NAME)
summay_wall, data_wall = data_tools.read_summary_data(data_path_walls, WALL_EXPERIMENT_NAME)
wall_data_tuple = (summay_wall, data_wall)

iid_path = os.path.join(EXPERIMENT_FOLDER, "iid.npy")
pol_path = os.path.join(EXPERIMENT_FOLDER, "pol.npy")

betas = [1.5]  # this is now beta
alphas = [0, 0.25, 1.5, 2.25, 4] #, 0.5, 0.75, 1] #, 3, 5]
num_runs = 5

num_clus_matrix = np.zeros((len(alphas), num_runs))
acc_matrix_final = np.zeros((len(alphas), num_runs))
acc_matrix_final_std = np.zeros_like(acc_matrix_final)
pol_over_wall_dist = np.zeros((len(alphas), num_runs, 400, 2))
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

# num_clus_matrix = np.zeros((num_runs, len(alphas), len(betas)))
# num_clus_matrix_std = np.zeros((num_runs, len(alphas), len(betas)))
# ord_matrix_final = np.zeros((num_runs, len(alphas), len(betas)))
# ord_matrix_final_std = np.zeros((num_runs, len(alphas), len(betas)))
# pol_matrix_final = np.zeros((num_runs, len(alphas), len(betas)))
# pol_matrix_final_std = np.zeros((num_runs, len(alphas), len(betas)))

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

            time_w = 25  # in minutes
            time_lim = num_t - (time_w * 60 * 30)
            if time_lim < 0:
                time_lim = 0
            print("Time lim: ", time_lim)
            valid_ts = [t for t in valid_ts if t > time_lim]
            valid_ts = valid_ts[0:-1]

            cluster_dict = data_tools.subgroup_clustering(summary, pm, iidm, valid_ts=np.array(valid_ts),
                                                          runi=runi)
            a = alphas.index(alpha)
            ri = runi
            comv_matrix_final[a, ri] = np.mean(com_vel[0, valid_ts])
            comv_matrix_final_std[a, ri] = np.std(com_vel[0, valid_ts])
            ord_matrix_final[a, ri] = np.mean(ord[0, valid_ts])
            ord_matrix_final_std[a, ri] = np.std(ord[0, valid_ts])
            pol_matrix_final[a, ri] = np.mean(mean_pol_vals_long[valid_ts])
            pol_matrix_final_std[a, ri] = np.std(mean_pol_vals_long[valid_ts])
            iid_matrix_final[a, ri] = np.mean(mean_iid_long[valid_ts])
            iid_matrix_final_std[a, ri] = np.std(mean_iid_long[valid_ts])
            coll_times[0, a, ri] = len(wall_reflection_times) / num_t
            coll_times[1, a, ri] = len(agent_reflection_times) / num_t
            abs_vel_m_final[a, ri] = abs_vel[..., valid_ts].mean(axis=-1)[0].mean()
            abs_vel_m_final_std[a, ri] = abs_vel[..., valid_ts].mean(axis=-1)[0].std()
            turn_rate_m_final[a, ri] = turning_rates[..., valid_ts].mean(axis=-1)[0].mean()
            turn_rate_final_std[a, ri] = turning_rates[..., valid_ts].mean(axis=-1)[0].std()


            # ord_matrix_final[runi, alphas.index(alpha), betas.index(beta)] = np.mean(ord[runi, valid_ts])
            # ord_matrix_final_std[runi, alphas.index(alpha), betas.index(beta)] = np.std(ord[runi, valid_ts])
            # pol_matrix_final[runi, alphas.index(alpha), betas.index(beta)] = np.mean(mean_pol_vals_long[valid_ts])
            # pol_matrix_final_std[runi, alphas.index(alpha), betas.index(beta)] = np.std(mean_pol_vals_long[valid_ts])
            # num_clus_matrix[runi, alphas.index(alpha), betas.index(beta)] = np.array(np.mean(cluster_dict["num_subgroups"]))
            # num_clus_matrix_std[runi, alphas.index(alpha), betas.index(beta)] = np.array(np.std(cluster_dict["num_subgroups"]))

            donei += 1
            print(f"Process: {donei/num_data_points*100}%")


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

show_change = "Alpha"


# plt.figure()
# plt.imshow(num_clus_matrix.T)

mean_clus = num_clus_matrix.mean(axis=1)
std_clus = num_clus_matrix.std(axis=1)
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
plt.imshow(num_clus_matrix.T)
plt.title("Mean number of subgroups")
plt.xticks([i for i in range(len(alphas))], alphas)
plt.yticks([i for i in range(num_runs)])
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