"""This script contains the main softeare tools to read back
data exported by the exploration tool (written in Julia) to explore tracking
data of the visual swarm experiments."""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

def float_to_str_without_points(float_number):
    """Converts a float number to string without points."""
    return str(float_number).replace('.', '')


def get_swarmviz_data(input_path):
    """Reading files exported from the SwarmViz tool for a single measurement and returning dataframes for robots and metrics."""

    # Reading back robot data from robots.parquet
    robots = pd.read_parquet(os.path.join(input_path, 'agents.parquet'), engine='pyarrow')

    # defining time axis where robot data is concatenated so has num_robots * T rows
    t = robots['t']

    # number of robots
    num_robots = np.max(robots['agent_id'])

    # undersampling time by number of robots to get real time
    t_u = t[::num_robots]

    # Reading metrics data
    metrics = pd.read_parquet(os.path.join(input_path, 'metrics.parquet'), engine='pyarrow')

    # exploring data that have been reead back
    print(metrics.head())
    print(metrics.columns)
    print(metrics.shape)
    print(metrics.dtypes)

    mean_pol = np.mean(metrics['Polarization'])
    mean_pol_std = np.std(metrics['Polarization'])
    return mean_pol, mean_pol_std

    # # plotting all metrics in a sublots with title and y axis label
    # fig, ax = plt.subplots(5, 2, figsize=(10, 10), sharex=True)
    # # choosing some pastel colors as many as the number of metrics
    # # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    # colors = ['xkcd:light blue', 'xkcd:light orange', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple',
    #           'xkcd:light yellow', 'xkcd:light pink', 'xkcd:light brown', 'xkcd:light grey', 'xkcd:light teal']
    # for i, col in enumerate(metrics.columns):
    #     ax[i // 2, i % 2].plot(t_u, metrics[col], color=colors[i])
    #     ax[i // 2, i % 2].set_title(col)
    #     ax[i // 2, i % 2].set_ylabel(col)
    #     plt.xlabel('time (s)')
    # plt.show()

    # # exploring data that have been reead back
    # print(robots.head())
    # print(robots.columns)
    # print(robots.shape)
    # print(robots.dtypes)
    #
    # # fixing agent_ids to be repetition of 0-10 T times
    # agent_ids_list = [i + 1 for i in range(num_robots)] * len(t_u)
    # print(t.shape, num_robots, robots['agent_id'].shape, len(agent_ids_list))
    # robots['agent_id'] = agent_ids_list

    #
    # # plotting all metrics in a sublots with title and y axis label
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
    # # choosing 20 pastel colors as many as the number of metrics
    # # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    # colors = ['xkcd:light blue', 'xkcd:light orange', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple',
    #           'xkcd:light yellow', 'xkcd:light pink', 'xkcd:light brown', 'xkcd:light blue', 'xkcd:light orange',
    #           'xkcd:light green', 'xkcd:light red', 'xkcd:light purple', 'xkcd:light yellow', 'xkcd:light pink',
    #           'xkcd:light brown', 'xkcd:light blue', 'xkcd:light orange', 'xkcd:light green']
    # for i, col in enumerate(robots.columns):
    #     ax[i // 4, i % 4].plot(t, robots[col], color=colors[i])
    #     ax[i // 4, i % 4].set_title(col)
    #     ax[i // 4, i % 4].set_ylabel(col)
    #     plt.xlabel('time (s)')
    # plt.show()

    # # Reading derived metrics coming from julia with HDF5
    # f = h5py.File(os.path.join(input_path, 'derived.jld2'), 'r')
    # print(list(f.keys()))
    # # print(f['center_of_mass'].shape)
    # # print(f['center_of_mass'].dtype)
    # # print(f['distance_matrices'].shape)
    # # print(f['distance_matrices'].dtype)
    # # print(f['furthest_agents'].shape)
    # # print(f['furthest_agents'].dtype)
    #
    # # defining center of mass
    # COM_x = f['center_of_mass'][:, 0]
    # COM_y = f['center_of_mass'][:, 1]
    #
    # # calculate mean distance:
    # mean_dist = np.mean(np.mean(f['distance_matrices'], axis=-1), axis=-1)
    # # calculate distance std
    # std_dist = np.mean(np.std(f['distance_matrices'], axis=-1), axis=-1)


data_path = "D:\\VSWRM Robot Data"
experiment_case = "A0Fixed"

data_path = os.path.join(data_path, experiment_case)
if experiment_case == "A0Fixed":
    # defining parameter combinations
    betas = [0, 0.1, 0.25, 0.75, 1.25, 1.75, 2.5, 4]
    alphas = [0.9]
    repetitions = [1, 2]

    # generating folder names and paths for the experiments
    param_combis = [(alpha, beta, repetition) for alpha in alphas for beta in betas for repetition in repetitions]
    experiment_base_names = [f"EXP1_A0_{float_to_str_without_points(alpha)}_B0_{float_to_str_without_points(beta)}_r{repetition}" for alpha in alphas for beta in betas for repetition in repetitions]
    final_experiment_files = []
    walls = []

    # get all subdirectories of data_path
    print([f.path for f in os.scandir(data_path) if f.is_dir()])
    subdirs = [f.path for f in os.scandir(data_path) if f.is_dir()]

    # matching subdirectories with measurement base names
    for experiment_base_name in experiment_base_names:
        for subdir in subdirs:
            if subdir.find(experiment_base_name)>-1:
                final_experiment_files.append(subdir)
                walls.append(subdir.split('_w')[1].split('.')[0])


mpols = np.zeros((len(alphas), len(betas), len(repetitions)))
mpols_std = np.zeros((len(alphas), len(betas), len(repetitions)))

for alpha in alphas:
    for beta in betas:
        for repetition in repetitions:
            index = param_combis.index((alpha, beta, repetition))
            print(f"Processing {experiment_base_names[index]} with walls {walls[index]}")
            input_path = os.path.join(final_experiment_files[index], "swarmvizexport")
            mean_pol, mean_pol_std = get_swarmviz_data(input_path)
            mpols[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = mean_pol
            mpols_std[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = mean_pol_std

# average over repetitions
mpols = np.mean(mpols, axis=-1)
mpols_std = np.mean(mpols_std, axis=-1)

# plotting mean polarizations with shaded background showing std
for i, alpha in enumerate(alphas):
    plt.plot([i for i in range(len(betas))], mpols[i], label=f"alpha: {alpha}")
    plt.fill_between([i for i in range(len(betas))], mpols[i] - mpols_std[i], mpols[i] + mpols_std[i], alpha=0.2)
    # equidistant ticks for beta
    plt.xticks([i for i in range(len(betas))], betas)
    # logaritmic scale for beta
    plt.ylim(0.45, 1)

plt.xlabel("beta")
plt.ylabel("mean polarization")
plt.legend()
plt.show()




    




