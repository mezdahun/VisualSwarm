"""This script contains the main softeare tools to read back
data exported by the exploration tool (written in Julia) to explore tracking
data of the visual swarm experiments."""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from visualswarm.simulation_tools import data_tools, plotting_tools


def float_to_str_without_points(float_number):
    """Converts a float number to string without points."""
    return str(float_number).replace('.', '')


def transform_table_to_groups(merge_operations, merge_heights, threshold):
    # Initialize a dictionary to map each robot or subgroup to its current subgroup
    # print(merge_operations)
    current_group = {}
    # Initialize a counter for new subgroup IDs
    new_group_id = 1

    for mi, merge in enumerate(merge_operations):
        if merge_heights[mi] < threshold:
            a = min(merge)  # a is robot/subgroup being merged, b is the target robot/subgroup
            b = max(merge)

            # If both a and b are robots (negative values), create a new subgroup
            if a < 0 and b < 0:
                # print(f"Creating new subgroup with robots {a} and {b}")
                current_group[a] = new_group_id
                current_group[b] = new_group_id
                # print(f"New subgroup created: {new_group_id}")
                # print(f"Current group: {current_group}")
                new_group_id += 1

            # If a is a robot and b is a subgroup, add a to subgroup b
            elif a < 0 and b > 0:
                # print(f"Adding robot {a} to subgroup {b}")
                current_group[a] = b
                # then iterate through all elements belonging to b and update their group id
                for robot, group_id in current_group.items():
                    if group_id == b:
                        current_group[robot] = new_group_id
                new_group_id += 1
                # print(f"Current group: {current_group}")

            # If a and b are both subgroups, merge them
            elif a > 0 and b > 0:
                # print(f"Merging subgroups {a} and {b}")
                # Find all elements belonging to a or be and update their group id
                for robot, group_id in current_group.items():
                    if group_id == a or group_id == b:
                        current_group[robot] = new_group_id
                new_group_id += 1
                # print(f"Current group: {current_group}")

    # Generate the final list of group IDs for robots, assuming robots are numbered from -1 downwards
    # and adjusting subgroup IDs to start from 0
    # number of robots
    max_robot_id = abs(np.min(merge_operations))
    # max_robot_id = abs(min(current_group.keys()))
    group_ids = [None] * max_robot_id
    for robot_id, group_id in current_group.items():
        if robot_id < 0:  # Ensure it's a robot
            group_ids[abs(robot_id) - 1] = group_id  # Adjust for 0-based indexing

    # replacing None with -1
    group_ids = [-1 if gi is None else gi for gi in group_ids]

    # translating ids to start from 0
    # create dictionary to map old group ids to new group ids
    group_id_map = {old_id: new_id for new_id, old_id in enumerate(set(group_ids))}
    group_ids = [group_id_map[gi] for gi in group_ids]

    return group_ids


def count_t_in_dict_value_lists(dict, t):
    """Count how many times t is in the values of a dictionary."""
    print(t)
    return sum(t in v for v in dict.values())


def get_swarmviz_data(input_path, metric=None, filtered_timesteps=None):
    """Reading files exported from the SwarmViz tool for a single measurement and returning dataframes for robots and metrics."""

    if metric is None:
        raise ValueError("Please provide a metric to read back.")
    elif metric not in ["NumSubgroups", "LargestCluster"]:
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

        pol = metrics[metric]
        # filtering time series according to filtered_timesteps if provided
        if filtered_timesteps is not None:
            pol = pol[~pol.index.isin(filtered_timesteps)]
        mean_pol = np.mean(pol)
        mean_pol_std = np.std(pol)

        return pol, mean_pol, mean_pol_std

    else:
        # Reading derived metrics coming from julia with HDF5
        f = h5py.File

        # check if clustering processed file already exists
        if os.path.exists(os.path.join(input_path, 'clustering.npy')):
            print("Clustering file already exists.")
            # clustering has shape T x num_robots
            clustering = np.load(os.path.join(input_path, 'clustering.npy'))

        else:
            print("Clustering file does not exist. Processing clustering.")
            f = h5py.File(os.path.join(input_path, 'derived.jld2'), 'r')
            T = f['clustering']['clustering_merges'].shape[0]
            robots = pd.read_parquet(os.path.join(input_path, 'agents.parquet'), engine='pyarrow')
            num_robots = np.max(robots['agent_id'])
            clustering = np.zeros((T, num_robots))
            for t in range(T):
                try:
                    print(f"{t / T * 100:.2f}%", end='\r', flush=True)
                    merges = f['clustering']['clustering_merges'][t, :, :]
                    heights = f['clustering']['clustering_heights'][t, :]
                    threshold = f['clustering']['chosen_clustering_threshold'][()]
                    group_ids = transform_table_to_groups(merges.T, heights, threshold)
                    clustering[t, :] = np.array(group_ids)
                except ValueError:
                    print(f"Error at time {t}")
                    print(merges)
                    print(heights)
                    print(threshold)
                    print(group_ids)
                    break

            np.save(os.path.join(input_path, 'clustering.npy'), clustering)

        if metric == "NumSubgroups":
            # measuring number of clusters
            pol = [len(set(clustering[t, :])) for t in range(clustering.shape[0])]
            mean_pol = np.mean(pol)
            mean_pol_std = np.std(pol)
            return pol, mean_pol, mean_pol_std

        elif metric == "LargestCluster":
            # measuring largets clusters
            pol = [np.max(np.bincount(clustering[t, :].astype(int))) for t in range(clustering.shape[0])]
            mean_pol = np.mean(pol)
            mean_pol_std = np.std(pol)
            return pol, mean_pol, mean_pol_std

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

    # exploring data that have been reead back
    print(robots.head())
    print(robots.columns)
    print(robots.shape)
    print(robots.dtypes)
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
experiment_cases = ["A0Fixed", "B0Fixed"]
metrics = ["Polarization", "Mean IID", "LargestCluster",
           "NumRCollisions"]  # "Polarization", "Mean IID", "Max Min IID", "Rotational Order", "Area"]
denominators = [1, 1000, 1, 0.01]
units = ["", "m", "", "%"]
titles = {
    "Polarization": "Polarization",
    "Mean IID": "Mean I.I.D.",
    "NumSubgroups": "Number of Subgroups",
    "LargestCluster": "Size of Largest Cluster",
    "NumRCollisions": "Collision Time"
}
fontsize_xylabel = 19
fontsize_title = 22
fontsize_ticks = 17

num_exp_cases = len(experiment_cases)
num_metrics = len(metrics)

# creating a figure with subplots for each metric (rows) and experiment case (columns)
fig, ax = plt.subplots(num_exp_cases, num_metrics, figsize=(10, 10), sharex="row", sharey="col")

for ei, experiment_case in enumerate(experiment_cases):
    data_path_case = os.path.join(data_path, experiment_case)
    if experiment_case == "A0Fixed":
        # defining parameter combinations
        betas = [0, 0.1, 0.25, 0.75, 1.25, 1.75, 2.5, 4]
        alphas = [0.9]
        repetitions = [1, 2]
    elif experiment_case == "B0Fixed":
        betas = [1.25]
        alphas = [0, 0.1, 0.25, 0.5, 0.9, 1.25, 2]
        repetitions = [1, 2]

    # generating folder names and paths for the experiments
    param_combis = [(alpha, beta, repetition) for alpha in alphas for beta in betas for repetition in repetitions]
    experiment_base_names = [
        f"EXP1_A0_{float_to_str_without_points(alpha)}_B0_{float_to_str_without_points(beta)}_r{repetition}" for alpha
        in alphas for beta in betas for repetition in repetitions]
    final_experiment_files = []
    walls = []

    # get all subdirectories of data_path_case
    print([f.path for f in os.scandir(data_path_case) if f.is_dir()])
    subdirs = [f.path for f in os.scandir(data_path_case) if f.is_dir()]

    # matching subdirectories with measurement base names
    for experiment_base_name in experiment_base_names:
        for subdir in subdirs:
            if subdir.find(experiment_base_name) > -1:
                final_experiment_files.append(subdir)
                walls.append(subdir.split('_w')[1].split('.')[0])

    for mi, metric in enumerate(metrics):
        pols = np.zeros((len(alphas), len(betas), len(repetitions)))
        pols_std = np.zeros((len(alphas), len(betas), len(repetitions)))

        for alpha in alphas:
            for beta in betas:
                rep_pols = []
                for repetition in repetitions:

                    index = param_combis.index((alpha, beta, repetition))
                    print(f"Processing {experiment_base_names[index]} with walls {walls[index]}")
                    input_parent = final_experiment_files[index]
                    input_path = os.path.join(input_parent, "swarmvizexport")
                    # retreiving data
                    # splitting experiment file into path until last folder and last folder
                    # find filename of "..._summaryd.npy" file in input_path

                    # get data from VSWRM exports for agent collisions and filtering
                    input = glob.glob(os.path.join(input_parent, "*_summaryd.npy"))[0].split("_summaryd.npy")[0]
                    # find which wall was used
                    used_wall_file_index = input.split("_w")[1].split("_")[0]
                    print(input, used_wall_file_index)
                    WALL_EXPERIMENT_NAME = f"w{used_wall_file_index}"

                    # wall rott is parent folder of input_paranet
                    wall_root = os.path.dirname(input_parent)
                    print(wall_root)
                    csv_path_walls = os.path.join(wall_root, WALL_EXPERIMENT_NAME + ".csv")
                    print(f"Using wall file: ", csv_path_walls)
                    data_tools.optitrackcsv_to_VSWRM(csv_path_walls, skip_already_summed=True, dropna=False)
                    summary_wall, data_wall = data_tools.read_summary_data(wall_root, WALL_EXPERIMENT_NAME)
                    data_wall = data_tools.resample_wall_coordinates(summary_wall, data_wall, dxy=5, with_plot=False)
                    wall_data_tuple = (summary_wall, data_wall)

                    summary, data = data_tools.read_summary_data(input_parent, input)
                    wall_refl_dict, ag_refl_dict = data_tools.mine_reflection_times_drotY(data, summary, summary_wall,
                                                                                          data_wall)
                    wall_refl_dict = wall_refl_dict['0']
                    ag_refl_dict = ag_refl_dict['0']
                    # wall_refl_dict, ag_refl_dict are dictionaries where the keys are robot ids and the values are lists of reflection times
                    # collect all timesteps in which any of the robots reflected from the wall
                    wall_times = np.unique(np.concatenate(list(wall_refl_dict.values())))
                    # getting those timepoints where at least 2 robots collided
                    # double_wall_times = [t for t in wall_times if len([k for k, v in wall_refl_dict.items() if t in v])>1]
                    # double_wall_times = [t for t in wall_times if count_t_in_dict_value_lists(wall_refl_dict, t)>1]
                    # print(len(wall_times), len(double_wall_times))
                    ag_times = np.unique(np.concatenate(list(ag_refl_dict.values())))
                    print(len(wall_times), len(ag_times))

                    if metric != "NumRCollisions":
                        # get data from swarmviz exports
                        pol, mean_pol, mean_pol_std = get_swarmviz_data(input_path, metric=metric,
                                                                        filtered_timesteps=wall_times)
                        pols[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = mean_pol
                        pols_std[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = mean_pol_std
                    else:
                        T = data.shape[-1]
                        pol = np.zeros(T)
                        for i, t in enumerate(ag_times):
                            pol[i] = 1
                        mean_pol = np.mean(pol)
                        mean_pol_std = 0
                        pols[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = mean_pol
                        pols_std[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = mean_pol_std

                #     # in case we take the mean of all repetitions first
                #     pol, _, _ = get_swarmviz_data(input_path, metric=metric)
                #     rep_pols.append(pol.values)
                #
                # for repetition in repetitions:
                #     concat_pols = np.concatenate(rep_pols)
                #     pols[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = np.mean(np.mean(concat_pols))
                #     pols_std[alphas.index(alpha), betas.index(beta), repetitions.index(repetition)] = np.mean(np.std(concat_pols))

        # average over repetitions
        mpols = np.mean(pols, axis=-1) / denominators[mi]
        mpols_std = np.mean(pols_std, axis=-1) / denominators[mi]

        if num_metrics == 1:
            plt.axes(ax[ei])
        else:
            plt.axes(ax[ei, mi])

        if experiment_case == "A0Fixed":
            # choosing color and it's pastel version for mean and std
            color = 'xkcd:light blue'
            color_std = 'xkcd:light blue'
            if ei == 0:
                if units[mi] != "":
                    plt.ylabel(titles[metrics[mi]] + " [" + units[mi] + "]", fontsize=fontsize_xylabel)
                else:
                    plt.ylabel(titles[metrics[mi]], fontsize=fontsize_xylabel)
                plt.title(titles[metrics[mi]], fontsize=fontsize_title)
                plt.xlabel("$\\beta_0$", fontsize=fontsize_xylabel)
            # plotting mean polarizations with shaded background showing std
            for i, alpha in enumerate(alphas):
                for ri in range(len(repetitions)):
                    # dotted gray lines for each repetition
                    plt.plot([i for i in range(len(betas))], pols[i, :, ri] / denominators[mi], color='gray',
                             linestyle='dotted')
                plt.plot([i for i in range(len(betas))], mpols[i], color=color)
                plt.fill_between([i for i in range(len(betas))], mpols[i] - mpols_std[i], mpols[i] + mpols_std[i],
                                 alpha=0.2, color=color_std)
                # equidistant ticks for beta
                plt.xticks([i for i in range(len(betas))], betas, fontsize=fontsize_ticks, rotation=-45)
                # set y axis fontsize
                plt.yticks(fontsize=fontsize_ticks)
                # logaritmic scale for beta
                # plt.ylim(0.45, 1)
            # if mi==0:
            #     plt.ylabel("$\\alpha_0=0.9$ Fixed")

        elif experiment_case == "B0Fixed":
            # choosing color and it's pastel version for mean and std
            color = 'xkcd:light orange'
            color_std = 'xkcd:light orange'
            # if mi==0:
            #     plt.ylabel("$\\beta_0=1.25$ Fixed")
            # plotting mean polarizations with shaded background showing std
            for i, beta in enumerate(betas):
                for ri in range(len(repetitions)):
                    # dotted gray lines for each repetition
                    plt.plot([i for i in range(len(alphas))], pols[:, i, ri] / denominators[mi], color='gray',
                             linestyle='dotted')
                plt.plot([i for i in range(len(alphas))], mpols[:, i], color=color)
                plt.fill_between([i for i in range(len(alphas))], mpols[:, i] - mpols_std[:, i],
                                 mpols[:, i] + mpols_std[:, i], alpha=0.2, color=color_std)
                # equidistant ticks for alpha, with rotation 45 to the right
                plt.xticks([i for i in range(len(alphas))], alphas, fontsize=fontsize_ticks, rotation=-45)
                # logaritmic scale for beta
                # plt.ylim(0.45, 1)
            if units[mi] != "":
                plt.ylabel(titles[metrics[mi]] + " [" + units[mi] + "]", fontsize=fontsize_xylabel)
            else:
                plt.ylabel(titles[metrics[mi]], fontsize=fontsize_xylabel)
            plt.xlabel("$\\alpha_0$", fontsize=fontsize_xylabel)
            # set y axis fontsize
            plt.yticks(fontsize=fontsize_ticks)

        # if ei==0:
        #     plt.ylabel(titles[metric])

# adjusting left, right, wspace and hspace
plt.subplots_adjust(left=0.067, right=0.926, bottom=0.11, top=0.88, wspace=0.288, hspace=0.348)
plt.show()
