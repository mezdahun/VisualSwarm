"""EXPERIMENT DESCRIPTION: Name of Experiment

@description: Here comes the description of the experiment with webots"""

EXPERIMENT_NAME = "NameOfExperiment"

# if we need the distance of a robot from a given point
DISTANCE_REFERENCE = [0, 0, 0]

from visualswarm.simulation_tools import data_tools, plotting_tools

# if data is freshly created first summarize it into multidimensional array
data_path = f'path/to/data/foldr/{EXPERIMENT_NAME}'
data_tools.summarize_experiment(data_path, EXPERIMENT_NAME)

## Choose 1 (str) or more (list) comparison argument(s)
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

change_along = 'V0'
change_along_alias = '$v_0$'

# change_along = ['ALP1', 'BET1']
# change_along_alias = ['$\\alpha_1$', '$\\beta_1$']

# retreiving data
summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)

# plotting data
plotting_tools.plot_velocities(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
plotting_tools.plot_distances(summary, data, DISTANCE_REFERENCE, changed_along=change_along, changed_along_alias=change_along_alias)
plotting_tools.plot_orientation(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)