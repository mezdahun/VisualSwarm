"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT_NAMES = ["EQDistLarge1", "EQDistLarge2", "EQDistSmall1", "EQDistSmall2", "EQDistSmallBetLarge1", "EQDistSmallBetLarge2", "EQDistLargeBetLarge1", "EQDistLargeBetLarge2"]
DISTANCE_REFERENCE = [0, 0, 0]

from visualswarm.simulation_tools import data_tools, plotting_tools

polrats_over_exps = []
for expi in range(len(EXPERIMENT_NAMES)):
    # if data is freshly created first summarize it into multidimensional array
    data_tools.optitrackcsv_to_VSWRM(f"C:\\Users\\David\\Desktop\\VisualSwarm\\data\\optitrack\\{EXPERIMENT_NAMES[expi]}.csv")
    data_path = f'C:\\Users\\David\\Desktop\\VisualSwarm\\data\\optitrack\\'


    change_along = None
    change_along_alias = None

    # retreiving data
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAMES[expi])

    #
    # plotting_tools.plot_mean_COMvel_over_runs(summary, data, stdcolor="#FF9848")
    # regime_name = "STRONGPOLARIZEDLINE"
    # experiment_names = [f"{regime_name}_startNONpolarized_6Bots",
    #                     f"{regime_name}_startMIDpolarized_6Bots",
    #                     f"{regime_name}_startpolarized_6Bots"]
    # paths = [f'C:\\Users\\David\\Documents\\VisualSwarm\\controllers\\blank_controller\\simulation_data\\{i}' for i in experiment_names]
    # colors = ["#ffcccc", "#ffffb2", "#cce5cc"]
    # rad_limits = [1, 2, 3]
    # titles = [f"Initial max $\\Delta\\Phi={i}$ [rad]" for i in rad_limits]
    # plotting_tools.plot_COMvelocity_summary_perinit(paths, titles, colors, "Mean Center-of-mass velocity for different intial conditions in polarized line regime")

    #
    # center_of_mass = np.mean(data[:, :, [1, 2, 3], :], axis=1)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.scatter(data[0, :, 1, 100], data[0, :, 3, 100])
    # plt.scatter(center_of_mass[0, 0, 100], center_of_mass[0, 2, 100], s=10)
    # plt.show()

    # plotting data
    # plotting_tools.plot_velocities(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_distances(summary, data, DISTANCE_REFERENCE, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_orientation(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
    #plotting_tools.plot_iid(summary, data, 0)
    # #
    #
    # # min_iid = data_tools.calculate_min_iid(summary, data)
    #
    # plotting_tools.plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848')
    # plotting_tools.plot_mean_iid_over_runs(summary, data, stdcolor='#FF9848')
    # plotting_tools.plot_min_iid_over_runs(summary, data, stdcolor="#FF9848")
    # plotting_tools.plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848')
    #
    #
    # plotting_tools.plot_mean_ploarization(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_mean_iid(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_min_iid(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)

    #### LAST STEPS
    pol_matrix = data_tools.calculate_ploarization_matrix(summary, data)

    mean_pol = np.mean(np.mean(pol_matrix, axis=1), axis=1)
    pol_ratios = []
    for i in range(100):
        pol_ratio = np.count_nonzero(mean_pol>i*0.01) /mean_pol.shape[1]
        pol_ratios.append(pol_ratio)

    polrats_over_exps.append(np.array(pol_ratios))



fig, ax = plt.subplots(1, 2, figsize=[10, 5], sharey=True)
x = np.array([i*0.01 for i in range(100)])

large = np.vstack((polrats_over_exps[0], polrats_over_exps[1]))
mlarge = np.mean(large, axis=0)
stdlarge = np.std(large, axis=0)

lhalflinei = np.where(np.isclose(mlarge, 0.5, rtol=3e-02))[0][0]
lhalfline = x[lhalflinei]
ax[0].axvline(lhalfline, 0, mlarge[lhalflinei], label="$R^{pol}_{thr+}$=0.5", color="#89bcff", linestyle="--")
ax[0].axhline(mlarge[lhalflinei], 0, lhalfline, color="#89bcff", linestyle="--")

largebl = np.vstack((polrats_over_exps[6], polrats_over_exps[7]))
mlargebl = np.mean(largebl, axis=0)
stdlargebl = np.std(largebl, axis=0)

lblhalflinei = np.where(np.isclose(mlargebl, 0.5, rtol=3e-02))[0][0]
lblhalfline = x[lblhalflinei]
ax[1].axvline(lblhalfline, 0, mlargebl[lblhalflinei], label="$R^{pol}_{thr+}$=0.5", color="#89bcff", linestyle="--")
ax[1].axhline(mlargebl[lblhalflinei], 0, lblhalfline, color="#89bcff", linestyle="--")

small = np.vstack((polrats_over_exps[2], polrats_over_exps[3]))
msmall = np.mean(small, axis=0)
stdsmall = np.std(small, axis=0)

shalflinei = np.where(np.isclose(msmall, 0.5, rtol=3e-02))[0][0]
shalfline = x[shalflinei]
ax[0].axvline(shalfline, 0, msmall[shalflinei], label="$R^{pol}_{thr+}$=0.5", color="#ffcc89", linestyle="--", alpha=0.5)
ax[0].axhline(msmall[shalflinei], 0, shalfline, color="#ffcc89", linestyle="--", alpha=0.5)

smallbl = np.vstack((polrats_over_exps[4], polrats_over_exps[5]))
msmallbl = np.mean(small, axis=0)
stdsmallbl = np.std(small, axis=0)

sblhalflinei = np.where(np.isclose(msmallbl, 0.5, rtol=3e-02))[0][0]
sblhalfline = x[sblhalflinei]
ax[1].axvline(sblhalfline, 0, msmallbl[shalflinei], label="$R^{pol}_{thr+}$=0.5", color="#ffcc89", linestyle="--")
ax[1].axhline(msmallbl[shalflinei], 0, sblhalfline, color="#ffcc89", linestyle="--", alpha=0.5)

plt.axes(ax[0])
plt.title('Low $\\beta_0$')
#error_band = plt.fill_between(x, mlarge-stdlarge, mlarge+stdlarge, alpha=0.5, edgecolor="#ffbcd9", facecolor="#ffbcd9")
plt.plot(x, mlarge, label="$R^{pol}_{thr+}$ with $L_{eq}^L$", color="#89bcff", linewidth="3")

#error_band = plt.fill_between(x, msmall-stdsmall, msmall+stdsmall, alpha=0.5, edgecolor="#ffcc89", facecolor="#ffcc89")
plt.plot(x, msmall, label="$R^{pol}_{thr+}$ with $L_{eq}^S$", color="#ffcc89", linewidth="3")



plt.xlabel('Polarization Threshold [AU]')
plt.ylabel('Relative Time Above Threshold $R^{pol}_{thr+}$')

plt.axes(ax[1])
plt.title('High $\\beta_0$')
#error_band = plt.fill_between(x, mlargebl-stdlargebl, mlargebl+stdlargebl, alpha=0.5, edgecolor="#89bcff", facecolor="#89bcff")
plt.plot(x, mlargebl, label="$R^{pol}_{thr+}$ with $L_{eq}^L$", color="#89bcff", linewidth="3")

#error_band = plt.fill_between(x, msmallbl-stdsmallbl, msmallbl+stdsmallbl, alpha=0.5, edgecolor="#ffcc89", facecolor="#ffcc89")
plt.plot(x, msmallbl, label="$R^{pol}_{thr+}$ with $L_{eq}^S$", color="#ffcc89", linewidth="3")


plt.xlabel('Polarization Threshold [AU]')
plt.legend()

plt.show()