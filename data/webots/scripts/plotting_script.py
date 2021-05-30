"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
import numpy as np

EXPERIMENT_NAME = "TESTAUTO"
DISTANCE_REFERENCE = [0, 0, 0]

from visualswarm.simulation_tools import data_tools, plotting_tools

# if data is freshly created first summarize it into multidimensional array
data_path = f'C:\\Users\\David\\Documents\\VisualSwarm\\controllers\\blank_controller\\simulation_data\\{EXPERIMENT_NAME}'
data_tools.summarize_experiment(data_path, EXPERIMENT_NAME)


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

change_along = None
change_along_alias = None

# retreiving data
summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)
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
plotting_tools.plot_orientation(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
plotting_tools.plot_iid(summary, data, 2)
# #
#
# # min_iid = data_tools.calculate_min_iid(summary, data)
#
# plotting_tools.plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848')
# plotting_tools.plot_mean_iid_over_runs(summary, data, stdcolor='#FF9848')
plotting_tools.plot_min_iid_over_runs(summary, data, stdcolor="#FF9848")
plotting_tools.plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848')
#
#
# plotting_tools.plot_mean_ploarization(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
# plotting_tools.plot_mean_iid(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
# plotting_tools.plot_min_iid(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)