import os
import numpy as np
import matplotlib.pyplot as plt

summary_path = "/media/david/DMezeySCIoI/VSWRMData/Webots/Webetos_vs_reality/"
experiment = "E22"
webots_path = os.path.join(summary_path, f"{experiment}_calcsave_webots_turncorr_FOV05")
reality_path = os.path.join(summary_path, f"{experiment}_calcsave_reality")

if experiment == "E22":
    alphas = [0.75]#, 0.5, 0.75, 1, 1.25, 1.5] #, 3, 5]
    betas = [0.01, 0.1, 1, 6, 14] #, 0.5, 0.75, 1] #, 3, 5]
else:
    betas = [8]  # this is now beta
    alphas = [0, 0.25, 1.5, 2.25, 4]

# Reading data from reality summary
acc_matrix_final_r = np.load(os.path.join(reality_path, "accm.npy"))
acc_matrix_final_std_r = np.load(os.path.join(reality_path, "accmstd.npy"))
ord_matrix_final_r = np.load(os.path.join(reality_path, "ordm.npy"))
ord_matrix_final_std_r = np.load(os.path.join(reality_path, "ordmstd.npy"))
coll_times_r = np.load(os.path.join(reality_path, "colltm.npy"))
abs_vel_m_final_r = np.load(os.path.join(reality_path, "absvm.npy"))
abs_vel_m_final_std_r = np.load(os.path.join(reality_path, "absvmstd.npy"))
turn_rate_m_final_r = np.load(os.path.join(reality_path, "turnm.npy"))
turn_rate_final_std_r = np.load(os.path.join(reality_path, "turnmstd.npy"))

# Reading data from Webots summary
acc_matrix_final_w = np.load(os.path.join(webots_path, "accm.npy"))
acc_matrix_final_std_w = np.load(os.path.join(webots_path, "accmstd.npy"))
ord_matrix_final_w = np.load(os.path.join(webots_path, "ordm.npy"))
ord_matrix_final_std_w = np.load(os.path.join(webots_path, "ordmstd.npy"))
coll_times_w = np.load(os.path.join(webots_path, "colltm.npy"))
abs_vel_m_final_w = np.load(os.path.join(webots_path, "absvm.npy"))
abs_vel_m_final_std_w = np.load(os.path.join(webots_path, "absvmstd.npy"))
turn_rate_m_final_w = np.load(os.path.join(webots_path, "turnm.npy"))
turn_rate_final_std_w = np.load(os.path.join(webots_path, "turnmstd.npy"))

if experiment == "E22":
    show_change = "Beta"
    xlabel = f"$\\beta_0$"
    alphas = betas
else:
    xlabel = f"$\\alpha_0$"
    show_change = "Alpha"

# Showing orders
mean_ord_r = ord_matrix_final_r.mean(axis=1)
std_ord_r = ord_matrix_final_std_r.mean(axis=1)
mean_ord_w = ord_matrix_final_w.mean(axis=1)
std_ord_w = ord_matrix_final_std_w.mean(axis=1)
fig, ax = plt.subplots(1, 1)
plt.title("Order")
plt.plot(mean_ord_r, label="reality")
plt.fill_between([i for i in range(len(alphas))], mean_ord_r-std_ord_r,
                  mean_ord_r+std_ord_r, alpha=0.2)
plt.plot(mean_ord_w, label="webots")
plt.fill_between([i for i in range(len(alphas))], mean_ord_w-std_ord_w,
                  mean_ord_w+std_ord_w, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(xlabel)
plt.ylabel("order [AU]")
plt.legend()

# Showing absolute velocities
mean_av_r = abs_vel_m_final_r.mean(axis=1)
std_av_r = abs_vel_m_final_std_r.mean(axis=1)
mean_av_w = abs_vel_m_final_w.mean(axis=1)
std_av_w = abs_vel_m_final_std_w.mean(axis=1)
fig, ax = plt.subplots(1, 1)
plt.title("Absolute Velocity")
plt.plot(mean_av_r, label="reality")
plt.fill_between([i for i in range(len(alphas))], mean_av_r-std_av_r,
                  mean_av_r+std_av_r, alpha=0.2)
plt.plot(mean_av_w, label="webots")
plt.fill_between([i for i in range(len(alphas))], mean_av_w-std_av_w,
                  mean_av_w+std_av_w, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(xlabel)
plt.ylabel("velocity [mm/ts]")
plt.legend()

# Showing agent-agent collision times
mean_aac_r = coll_times_r[1].mean(axis=1)
std_aac_r = coll_times_r[1].std(axis=1)
mean_aac_w = coll_times_w[1].mean(axis=1)
std_aac_w = coll_times_w[1].std(axis=1)
fig, ax = plt.subplots(1, 1)
plt.title("Agent-agent collisions")
plt.plot(mean_aac_r, label="reality")
plt.fill_between([i for i in range(len(alphas))], mean_aac_r-std_aac_r,
                  mean_aac_r+std_aac_r, alpha=0.2)
plt.plot(mean_aac_w, label="webots")
plt.fill_between([i for i in range(len(alphas))], mean_aac_w-std_aac_w,
                  mean_aac_w+std_aac_w, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(xlabel)
plt.ylabel("relative ER time ratio")
plt.legend()

# Showing turning rates
mean_tr_r = turn_rate_m_final_r.mean(axis=1)
std_tr_r = turn_rate_final_std_r.mean(axis=1)
mean_tr_w = turn_rate_m_final_w.mean(axis=1)
std_tr_w = turn_rate_final_std_w.mean(axis=1)
fig, ax = plt.subplots(1, 1)
plt.title("Turning Rate")
plt.plot(mean_tr_r, label="reality")
plt.fill_between([i for i in range(len(alphas))], mean_tr_r-std_tr_r,
                  mean_tr_r+std_tr_r, alpha=0.2)
plt.plot(mean_tr_w, label="webots")
plt.fill_between([i for i in range(len(alphas))], mean_tr_w-std_tr_w,
                  mean_tr_w+std_tr_w, alpha=0.2)
plt.xticks([i for i in range(len(alphas))], alphas)
plt.xlabel(xlabel)
plt.ylabel("Turning rate [mm/ts]")
plt.legend()

# # Showing IID
# mean_iid_r = iid_matrix_final_r.mean(axis=1)
# std_iid_r = iid_matrix_final_std_r.mean(axis=1)
# mean_iid_w = iid_matrix_final_w.mean(axis=1)
# std_iid_w = iid_matrix_final_std_w.mean(axis=1)
# fig, ax = plt.subplots(1, 2)
# plt.axes(ax[0])
# plt.imshow(iid_matrix_final.T)
# plt.title("Mean IID")
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.yticks([i for i in range(num_runs)])
# plt.ylabel("runs")
# plt.xlabel(f"${show_change}_0$")
# plt.axes(ax[1])
# plt.plot(mean_iid)
# plt.fill_between([i for i in range(len(alphas))], mean_iid-std_iid,
#                   mean_iid+std_iid, alpha=0.2)
# plt.xticks([i for i in range(len(alphas))], alphas)
# plt.xlabel(f"${show_change}_0$")
# plt.ylabel("mean IID [mm]")
# plt.legend()

plt.show()