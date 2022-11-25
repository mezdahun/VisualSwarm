"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
import numpy as np

EXPERIMENT_NAMES = ["EQDistSmall1"]#, "EQDistSmall2", "EQDistLarge1", "EQDistLarge2"]#, "EQDistLarge2", "EQDistLargeBetLarge1", "EQDistLargeBetLarge2"]
DISTANCE_REFERENCE = [0, 0, 0]

from visualswarm.simulation_tools import data_tools, plotting_tools

polrats_over_exps = []
for expi in range(len(EXPERIMENT_NAMES)):
    # if data is freshly created first summarize it into multidimensional array
    data_tools.optitrackcsv_to_VSWRM(f"C:\\Users\\David\\Desktop\\VisualSwarm\\data\\optitrack\\{EXPERIMENT_NAMES[expi]}.csv")
    data_path = f'C:\\Users\\David\\Desktop\\VisualSwarm\\data\\optitrack\\'
    # data_tools.summarize_experiment(data_path, EXPERIMENT_NAME)


    # change_along = None
    # change_along_alias = None

    # change_along = 'KAP'
    # change_along_alias = '$\kappa$'

    # change_along = 'GAM'
    # change_along_alias = '$\gamma$'

    # change_along = 'ALP0'
    # change_along_alias = '$\\alpha_0$'

    # change_along = 'ALP1'
    # change_along_alias = '$\\alpha_1$'

    # change_along = 'BET0'
    # change_along_alias = '$\\beta_0$'

    # change_along = 'V0'
    # change_along_alias = '$v_0$'

    # change_along = ['ALP1', 'BET1']
    # change_along_alias = ['$\\alpha_1$', '$\\beta_1$']

    import matplotlib.pyplot as plt
    change_along = None
    change_along_alias = None

    # retreiving data
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAMES[expi])
    data[0, 0, 0, :]*=1000
    #
    fig, ax = plt.subplots(2, 1, figsize=[10, 6], sharex=True)
    plotting_tools.plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848', ax=ax[0])
    plotting_tools.plot_mean_iid_over_runs(summary, data, stdcolor='#FF9848', ax=ax[1])

    plt.axes(ax[0])
    plt.ylabel('Mean Polarization [AU]')

    plt.axes(ax[1])
    plt.ylabel('Mean i.i.d [mm]')
    plt.axhline(740)
    plt.xlabel('Time [s]')

    plt.show()

    #plotting_tools.plot_min_iid_over_runs(summary, data, stdcolor="#FF9848")
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
    # pol_matrix = data_tools.calculate_ploarization_matrix(summary, data)
    # print(pol_matrix.shape)
    #
    # mean_pol = np.mean(np.mean(pol_matrix, axis=1), axis=1)
    # pol_ratios = []
    # for i in range(100):
    #     pol_ratio = np.count_nonzero(mean_pol>i*0.01) /mean_pol.shape[1]
    #     pol_ratios.append(pol_ratio)
    # import matplotlib.pyplot as plt
    # plt.plot(pol_ratios, label=f"{EXPERIMENT_NAMES[expi]}")
    # polrats_over_exps.append(np.array(pol_ratios))


#     from peakutils.baseline import baseline
#     iid = data_tools.calculate_interindividual_distance(summary, data)
#     print(iid.shape)
#     mean_iid = np.mean(np.mean(iid, axis=1), axis=1)
#     bl = baseline(mean_iid[0,::3])
#     plt.plot(mean_iid[0,::3])
#     print(np.mean(mean_iid[mean_iid<2000]))
#     # plt.plot(bl)
#     plt.show()
#
#
#
#
# # plt.legend()
# # plt.figure()
# # large = np.vstack((polrats_over_exps[0], polrats_over_exps[1]))
# # print(large.shape)
# # mlarge = np.mean(large, axis=0)
# # small = np.vstack((polrats_over_exps[2], polrats_over_exps[3]))
# # msmall = np.mean(small, axis=0)
# # plt.plot(mlarge, label="large eq dist")
# # plt.plot(msmall, label="small eq dist")
# # plt.show()
