"""
@author: mezdahun
@description: tools to plot recorded values in webots (only works with saving data)
"""
import os
import time

from visualswarm.simulation_tools import data_tools
import matplotlib.patches as patches
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.cluster.hierarchy import linkage

def draw_line(x,y,angle,length):
  terminus_x = x + length * math.cos(angle)
  terminus_y = y + length * math.sin(angle)
  print([x, terminus_x],[y,terminus_y])
  plt.plot([x, terminus_x],[y,terminus_y])


def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage

def plot_replay_run(summary, data, runi=0, t_start=0, t_end=None, t_step=None, step_by_step=False,
                    x_min=-5000, x_max=5000, wall_coordinates=None, history_length=0):
    """Replaying experiment from summary and data in matplotlib plot"""
    if wall_coordinates is not None:
        x_min = np.nanmin(wall_coordinates[0, :]) - 100
        x_max = np.nanmax(wall_coordinates[0, :]) + 100
        y_min = np.nanmin(wall_coordinates[1, :]) - 100
        y_max = np.nanmax(wall_coordinates[1, :]) + 100
        print(x_min, x_max, y_min, y_max)
    else:
        y_min = x_min
        y_max = x_max

    num_robots = data.shape[1]

    iidm = data_tools.calculate_interindividual_distance(summary, data)[runi, ...]
    pm = data_tools.calculate_ploarization_matrix(summary, data)

    if t_end is None:
        t_end = data.shape[-1]
    if t_step is None:
        t_step = 100

    plt.ion()
    fig, ax = plt.subplots(2, 1)
    for t in range(t_start, t_end, t_step):

        if num_robots > 1:
            plt.axes(ax[0])
            niidm = (iidm[:, :, t] - np.min(iidm[:, :, t])) / (np.max(iidm[:, :, t]) - np.min(iidm[:, :, t]))
            dist = (1 - pm[runi, :, :, t].astype('float') + niidm) / 2
            # sermat = compute_serial_matrix(1-pm[0, :, :, t].astype('float'))
            linkage_matrix = linkage(dist, "single")
            ret = dendrogram(linkage_matrix, color_threshold=1.2, labels=[i for i in range(num_robots)], show_leaf_counts=True)

        plt.axes(ax[1])
        center_of_mass = np.mean(data[:, :, [1, 2, 3], :], axis=1)
        if num_robots > 1:
            colors = [color for _, color in sorted(zip(ret['leaves'], ret['leaves_color_list']))]
        else:
            colors = "blue"

        plt.scatter(data[runi, :, 1, t], data[runi, :, 3, t], s=100, c=colors)

        for i in range(num_robots):
            plt.annotate(i+1, (data[runi, i, 1, t], data[runi, i, 3, t] + 0.2))

        ori = data[runi, :, 4, t]
        ms = 200
        for ri in range(len(ori)):
            x = data[runi, :, 1, t]
            y = data[runi, :, 3, t]
            angle = ori[ri]
            plt.arrow(x[ri], y[ri], ms * math.cos(angle), ms * math.sin(angle), color="white")

        plt.scatter(center_of_mass[runi, 0, t], center_of_mass[runi, 2, t], s=50)

        if history_length > 0:
            for robi in range(num_robots):
                if isinstance(colors, list):
                    col = colors[robi]
                else:
                    col = "blue"
                plt.plot(data[runi, robi, 1, t-history_length:t:5], data[runi, robi, 3, t-history_length:t:5], '-', c=col)

        if wall_coordinates is not None:
            plt.plot(wall_coordinates[0, :], wall_coordinates[1, :], '--', c="black")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.draw()
        if step_by_step:
            input()
        else:
            fig.canvas.draw()
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.05)
        plt.clf()

def plot_velocities(summary, data, changed_along=None, changed_along_alias=None):
    velocities = data_tools.calculate_velocity(summary, data)
    fig, ax = plt.subplots(summary['num_robots'], 1, figsize=[10, 8])

    time = data[0, 0, 0, :-1]  # 1 step shorter because of diff

    if changed_along is not None:
        if not isinstance(changed_along, list):
            changed_along = [changed_along]

    if changed_along_alias is not None:
        if not isinstance(changed_along_alias, list):
            changed_along_alias = [changed_along_alias]

    for i in range(summary['num_robots']):
        if summary['num_robots'] > 1:
            plt.axes(ax[i])

        plt.plot(time, velocities[:, i, :].T)

        if changed_along is not None:
            if changed_along_alias is None:
                legend = [', '.join([f'{ca}={summary["params"][f"run{j+1}"][ca]}' for ca in changed_along]) for j in range(summary['num_runs'])]
            else:
                legend = [', '.join([f'{changed_along_alias[i]}={summary["params"][f"run{j + 1}"][changed_along[i]]}' for i in range(len(changed_along))]) for j in
                          range(summary['num_runs'])]
            plt.legend(legend)

        plt.title(f'Velocity of robot {i}')
        plt.xlabel('simulation time [s]')
        plt.ylabel('velocity [m/s]')

    plt.show()


def plot_distances(summary, data, reference_point, changed_along=None, changed_along_alias=None):
    distances = data_tools.calculate_distance(summary, data, reference_point)
    fig, ax = plt.subplots(summary['num_robots'], 1, figsize=[10, 8])

    time = data[0, 0, 0, :]  # 1 step shorter because of diff

    if changed_along is not None:
        if not isinstance(changed_along, list):
            changed_along = [changed_along]

    if changed_along_alias is not None:
        if not isinstance(changed_along_alias, list):
            changed_along_alias = [changed_along_alias]

    for i in range(summary['num_robots']):
        if summary['num_robots'] > 1:
            plt.axes(ax[i])
        plt.plot(time, distances[:, i, :].T)

        if changed_along is not None:
            if changed_along_alias is None:
                legend = [', '.join([f'{ca}={summary["params"][f"run{j+1}"][ca]}' for ca in changed_along]) for j in range(summary['num_runs'])]
            else:
                legend = [', '.join([f'{changed_along_alias[i]}={summary["params"][f"run{j + 1}"][changed_along[i]]}' for i in range(len(changed_along))]) for j in
                          range(summary['num_runs'])]
            plt.legend(legend)

        plt.title(f'Distance of robot {i} from point {reference_point}')
        plt.xlabel('simulation time [s]')
        plt.ylabel('distance [m]')

    plt.show()


def plot_orientation(summary, data, changed_along=None, changed_along_alias=None):

    fig, ax = plt.subplots(summary['num_robots'], 1, figsize=[10, 8])

    time = data[0, 0, 0, :]  # 1 step shorter because of diff

    or_idx = or_idx = summary['attributes'].index('or')
    orientations = data[:, :, or_idx, :]

    if changed_along is not None:
        if not isinstance(changed_along, list):
            changed_along = [changed_along]

    if changed_along_alias is not None:
        if not isinstance(changed_along_alias, list):
            changed_along_alias = [changed_along_alias]

    for i in range(summary['num_robots']):
        if summary['num_robots'] > 1:
            plt.axes(ax[i])
        plt.plot(time, orientations[:, i, :].T)

        if changed_along is not None:
            if changed_along_alias is None:
                legend = [', '.join([f'{ca}={summary["params"][f"run{j+1}"][ca]}' for ca in changed_along]) for j in range(summary['num_runs'])]
            else:
                legend = [', '.join([f'{changed_along_alias[i]}={summary["params"][f"run{j + 1}"][changed_along[i]]}' for i in range(len(changed_along))]) for j in
                          range(summary['num_runs'])]
            plt.legend(legend)

        plt.title(f'Orientation of robot {i}')
        plt.xlabel('simulation time [s]')
        plt.ylabel('orientation [rad]')

    plt.show()

def plot_iid(summary, data, run_id):
    iid = data_tools.calculate_interindividual_distance(summary, data)
    fig, ax = plt.subplots(summary['num_robots'], summary['num_robots'], figsize=[10, 10], sharex=True, sharey=True)

    time = data[0, 0, 0, :]

    for i in range(summary['num_robots']):
        for j in range(i + 1):
            if summary['num_robots'] > 1:
                plt.axes(ax[i, j])
            plt.plot(time, iid[run_id, i, j, :])
            if i == summary['num_robots']-1:
                plt.xlabel('time [s]')
            if j == 0:
                plt.ylabel('ii distance [m]')

    plt.suptitle(f'Interindividual distances in run {run_id}')

    plt.show()

def plot_mean_ploarization(summary, data, changed_along=None, changed_along_alias=None):
    x_axis = []
    mean_pol = data_tools.calculate_mean_polarization(summary, data)

    if changed_along is not None:
        for i in range(summary['num_runs']):
            x_axis.append(summary['params'][f'run{i+1}'][changed_along])

    else:
        x_axis = [i+1 for i in range(summary['num_runs'])]

    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    plt.plot(x_axis, mean_pol)

    if changed_along is not None:
        plt.title(f'Mean polarization for changing {changed_along_alias}')
        plt.xlabel(changed_along_alias)
        plt.ylabel('Normalized Mean Polarization AU$\\in$[-1,1]')
    else:
        plt.title(f'Mean polarization of robots')
        plt.xlabel('run number')
        plt.ylabel('Normalized Mean Polarization AU$\\in$[-1,1]')

    plt.show()

def plot_mean_iid(summary, data, changed_along=None, changed_along_alias=None):
    x_axis = []
    mean_iid = data_tools.calculate_mean_iid(summary, data)

    if changed_along is not None:
        for i in range(summary['num_runs']):
            x_axis.append(summary['params'][f'run{i+1}'][changed_along])

    else:
        x_axis = [i+1 for i in range(summary['num_runs'])]

    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    plt.plot(x_axis, mean_iid)

    if changed_along is not None:
        plt.title(f'Mean IID for changing {changed_along_alias}')
        plt.xlabel(changed_along_alias)
        plt.ylabel('Mean IID [m]')
    else:
        plt.title(f'Mean IID of robots')
        plt.xlabel('run number')
        plt.ylabel('Mean IID [m]')

    plt.show()

def plot_min_iid(summary, data, changed_along=None, changed_along_alias=None):
    x_axis = []
    min_iid = data_tools.calculate_min_iid(summary, data)

    if changed_along is not None:
        for i in range(summary['num_runs']):
            x_axis.append(summary['params'][f'run{i+1}'][changed_along])

    else:
        x_axis = [i+1 for i in range(summary['num_runs'])]

    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    plt.plot(x_axis, min_iid)

    if changed_along is not None:
        plt.title(f'Minimum IID for changing {changed_along_alias}')
        plt.xlabel(changed_along_alias)
        plt.ylabel('Min IID [m]')
    else:
        plt.title(f'Minimum IID of robots')
        plt.xlabel('run number')
        plt.ylabel('Min IID [m]')

    plt.show()

def plot_mean_iid_over_runs(summary, data, stdcolor='#FF9848', ax=None, with_legend=None):
    iid = data_tools.calculate_interindividual_distance(summary, data)

    population_mean = np.mean(np.mean(iid, 1), 1)
    run_mean = np.mean(population_mean, 0)
    run_std = np.std(population_mean, 0)
    t = data[0, 0, 0, :]

    if ax is not None:
        plt.axes(ax)

    for i in range(population_mean.shape[0]):
        individ_line, = plt.plot(t, population_mean[i, :], ls="--", color="gray", linewidth="0.2")
    mean_line, = plt.plot(t, run_mean, color="black", linewidth="1.2")
    error_band = plt.fill_between(t, run_mean - run_std, run_mean + run_std, alpha=0.5, edgecolor=stdcolor,
                                  facecolor=stdcolor)
    if ax is None:
        plt.title("Mean inter-individual distance")
        plt.xlabel("simulation time [s]")
        plt.ylabel("I.I.D [m]$>$BL=0.11")
        plt.legend([individ_line, mean_line, error_band], ["Single run population mean i.i.d.",
                                                           "Mean population i.i.d. over runs",
                                                           "Standard Deviation over runs"])
        plt.show()
    elif with_legend:
        plt.legend([individ_line, mean_line, error_band], ["Single run population mean i.i.d.",
                                                           "Mean population i.i.d. over runs",
                                                           "Standard Deviation over runs"])



def plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848', ax=None, with_legend=None):
    pol_m = data_tools.calculate_ploarization_matrix(summary, data)

    population_mean = np.mean(np.mean(pol_m, 1), 1)
    run_mean = np.mean(population_mean, 0)
    run_std = np.std(population_mean, 0)
    t = data[0, 0, 0, :]

    if ax is not None:
        plt.axes(ax)

    for i in range(population_mean.shape[0]):
        individ_line, = plt.plot(t, population_mean[i, :], ls="--", color="gray", linewidth="0.2")

    mean_line, = plt.plot(t, run_mean, color="black", linewidth="1.2")
    error_band = plt.fill_between(t, run_mean-run_std, run_mean+run_std, alpha=0.5, edgecolor=stdcolor, facecolor=stdcolor)

    if ax is None:
        plt.title("Population polarization (heading angle match)")
        plt.xlabel("simulation time [s]")
        plt.ylabel("polarization [AU]$\\in$[-1, 1]")
        plt.legend([individ_line, mean_line, error_band], ["Single run population avg polarization",
                                                           "Mean population polarization over runs",
                                                           "Standard Deviation over runs"])
        plt.show()
    elif with_legend:
        plt.legend([individ_line, mean_line, error_band], ["Single run population avg polarization",
                                                           "Mean population polarization over runs",
                                                           "Standard Deviation over runs"])

def plot_mean_pol_summary_perInit(paths, titles, colors, supertitle,
                                  xlabel="time [s]", ylabel="polariuzation [AU]$\\in$[0, 1]"):

    fig, axs = plt.subplots(len(paths), 1, figsize=[10, 8], sharex=True, sharey=True)

    for i in range(len(paths)):
        summary, data = data_tools.read_summary_data(paths[i], os.path.basename(paths[i]))
        plot_mean_pol_over_runs(summary, data, stdcolor=colors[i], ax=axs[i], with_legend=True)
        plt.ylabel(ylabel)
        plt.title(titles[i])

    plt.xlabel(xlabel)
    plt.suptitle(supertitle)
    plt.show()

def plot_mean_pol_summary_perRegimeandInit(regime_dict, titles, colors, supertitle,
                                           xlabel="time [s]", ylabel="polarization [AU]$\\in$[0, 1]"):

    fig, axs = plt.subplots(len(regime_dict), 1, figsize=[10, 8], sharex=True, sharey=True)

    reg_i = 0
    for regime_name, path_dict in regime_dict.items():
        paths = path_dict['paths']
        for i in range(len(paths)):
            summary, data = data_tools.read_summary_data(paths[i], os.path.basename(paths[i]))
            plot_mean_pol_over_runs(summary, data, stdcolor=colors[i], ax=axs[reg_i], with_legend=False)
            plt.ylabel(ylabel)
            plt.title(titles[reg_i])

        reg_i += 1

    plt.xlabel(xlabel)
    plt.suptitle(supertitle)
    plt.show()



def plot_min_iid_over_runs(summary, data, stdcolor="#FF9848", ax=None, with_legend=None):
    iid = data_tools.calculate_interindividual_distance(summary, data)
    for i in range(summary['num_robots']):
        iid[:, i, i, :] = np.inf

    population_mean = np.mean(np.min(iid, 1), 1)
    run_mean = np.mean(population_mean, 0)
    run_std = np.std(population_mean, 0)
    t = data[0, 0, 0, :]

    if ax is not None:
        plt.axes(ax)

    for i in range(population_mean.shape[0]):
        individ_line, = plt.plot(t, population_mean[i, :], ls="--", color="gray", linewidth="0.2")
    mean_line, = plt.plot(t, run_mean, color="black", linewidth="1.2")
    error_band = plt.fill_between(t, run_mean - run_std, run_mean + run_std, alpha=0.5, edgecolor=stdcolor,
                                  facecolor=stdcolor)

    if ax is None:
        plt.title("AVG of Minimum inter-individual distances over runs")
        plt.xlabel("simulation time [s]")
        plt.ylabel("min I.I.D [m]$>$BL=0.11")
        plt.legend([individ_line, mean_line, error_band], ["Min i.i.d. in single run",
                                                           "Mean of minimum i.i.ds over runs",
                                                           "Standard Deviation over runs"])
        plt.show()
    elif with_legend:
        plt.legend([individ_line, mean_line, error_band], ["Min i.i.d. in single run",
                                                           "Mean of minimum i.i.ds over runs",
                                                           "Standard Deviation over runs"])

def plot_min_iid_summary_perInit(paths, titles, colors, supertitle, xlabel="time [s]", ylabel="i-i.d. [m]"):

    fig, axs = plt.subplots(len(paths), 1, figsize=[10, 8], sharex=True, sharey=True)

    for i in range(len(paths)):
        summary, data = data_tools.read_summary_data(paths[i], os.path.basename(paths[i]))
        plot_min_iid_over_runs(summary, data, stdcolor=colors[i], ax=axs[i], with_legend=True)
        plt.ylabel(ylabel)
        plt.title(titles[i])

    plt.xlabel(xlabel)
    plt.suptitle(supertitle)
    plt.show()

def plot_min_iid_summary_perRegimeandInit(regime_dict, titles, colors, supertitle, xlabel="time [s]", ylabel="i-i d. [m]"):

    fig, axs = plt.subplots(len(regime_dict), 1, figsize=[10, 8], sharex=True, sharey=True)

    reg_i = 0
    for regime_name, path_dict in regime_dict.items():
        paths = path_dict['paths']
        for i in range(len(paths)):
            summary, data = data_tools.read_summary_data(paths[i], os.path.basename(paths[i]))
            plot_min_iid_over_runs(summary, data, stdcolor=colors[i], ax=axs[reg_i], with_legend=False)
            plt.ylabel(ylabel)
            plt.title(titles[reg_i])

        reg_i += 1

    plt.xlabel(xlabel)
    plt.suptitle(supertitle)
    plt.show()



def plot_mean_COMvel_over_runs(summary, data, stdcolor="#FF9848", ax=None, with_legend=None):
    COMvelocity = data_tools.population_velocity(summary, data)

    cut_beginning = 3

    run_mean = np.mean(COMvelocity, 0)[cut_beginning:]
    run_std = np.std(COMvelocity, 0)[cut_beginning:]

    # cutting initial spike
    t = data[0, 0, 0, cut_beginning:-1]

    if ax is not None:
        plt.axes(ax)

    for i in range(COMvelocity.shape[0]):
        individ_line, = plt.plot(t, COMvelocity[i, cut_beginning:], ls="--", color="gray", linewidth="0.2")
    mean_line, = plt.plot(t, run_mean, color="black", linewidth="1.2")
    error_band = plt.fill_between(t, run_mean - run_std, run_mean + run_std, alpha=0.5, edgecolor=stdcolor,
                                  facecolor=stdcolor)

    if ax is None:
        plt.title("Mean absolute velocity of Center of Mass")
        plt.xlabel("simulation time [s]")
        plt.ylabel("mean v [m/s]")
        plt.legend([individ_line, mean_line, error_band], ["Velocity of COM in individual runs",
                                                           "Mean of COM velocity over runs",
                                                           "Standard Deviation over runs"])

        plt.show()
    elif with_legend:
        plt.legend([individ_line, mean_line, error_band], ["Velocity of COM in individual runs",
                                                           "Mean of COM velocity over runs",
                                                           "Standard Deviation over runs"])

def plot_COMvelocity_summary_perInit(paths, titles, colors, supertitle, xlabel="time [s]", ylabel="v [m/s]"):

    fig, axs = plt.subplots(len(paths), 1, figsize=[10, 8], sharex=True, sharey=True)

    for i in range(len(paths)):
        summary, data = data_tools.read_summary_data(paths[i], os.path.basename(paths[i]))
        plot_mean_COMvel_over_runs(summary, data, stdcolor=colors[i], ax=axs[i], with_legend=True)
        plt.ylabel(ylabel)
        plt.title(titles[i])

    plt.xlabel(xlabel)
    plt.suptitle(supertitle)
    plt.show()

def plot_COMvelocity_summary_perRegimeandInit(regime_dict, titles, colors, supertitle, xlabel="time [s]", ylabel="v [m/s]"):

    fig, axs = plt.subplots(len(regime_dict), 1, figsize=[10, 8], sharex=True, sharey=True)

    reg_i = 0
    for regime_name, path_dict in regime_dict.items():
        paths = path_dict['paths']
        for i in range(len(paths)):
            summary, data = data_tools.read_summary_data(paths[i], os.path.basename(paths[i]))
            plot_mean_COMvel_over_runs(summary, data, stdcolor=colors[i], ax=axs[reg_i], with_legend=False)
            plt.ylabel(ylabel)
            plt.title(titles[reg_i])

        reg_i += 1

    plt.xlabel(xlabel)
    plt.suptitle(supertitle)
    plt.show()


def plot_reflection_effect_polarization(summary, data, ax=None):
    """Showing the effect of being reflected from walls in a single experiment with multiple runs"""
    if ax is None:
        show_plot = True
        fig, ax = plt.subplots(1, summary['num_runs'], figsize=[10, 8])
    else:
        show_plot = False

    pol_m = data_tools.calculate_ploarization_matrix(summary, data)
    population_mean = np.mean(np.mean(pol_m, 1), 1)
    col_types = data_tools.get_collisions_with_type(summary, data)
    t = data[0, 0, 0, :]

    for i in range(summary['num_runs']):

        if summary['num_runs'] > 1:
            plt.axes(ax[i])
        else:
            plt.axes(ax)

        # plotting ploarization over runs
        N = 300
        t_ma = t[int(N/2)-1:int(-N/2)]
        ma = data_tools.moving_average(population_mean[i, :], N)
        # moving_avg_line, = plt.plot(t_ma, ma, color="black", linewidth="1.5")

        individ_line, = plt.plot(t, population_mean[i, :], color="black", linewidth="1.5")

        # showing collision times
        # if summary['num_runs'] > 1:
        for rob_i in range(summary['num_robots']):
            types = [elem[1] for elem in col_types[i][rob_i]]
            colors = ['red' if type == "withRobot" else 'gray' for type in types]
            col_times = [elem[0] / 1000 for elem in col_types[i][rob_i]]
            mask = [np.where(t == elem)[0][0] for elem in col_times]
            plt.scatter(t[mask], population_mean[i, mask], color=colors)

    if show_plot:
        plt.show()

def plot_reflection_effect_COMvelocity(summary, data, ax=None):
    """Showing the effect of being reflected from walls in a single experiment with multiple runs"""
    if ax is None:
        show_plot = True
        fig, ax = plt.subplots(1, summary['num_runs'], figsize=[10, 8])
    else:
        show_plot = False

    COMvelocity = data_tools.population_velocity(summary, data)
    col_types = data_tools.get_collisions_with_type(summary, data)

    cut_beginning = 3
    t = data[0, 0, 0, cut_beginning:-1]

    for i in range(summary['num_runs']):

        if summary['num_runs'] > 1:
            plt.axes(ax[i])
        else:
            plt.axes(ax)

        # plotting ploarization over runs
        # print('SHAPES')
        N = 300
        t_ma = t[int(N/2)-1:int(-N/2)]
        ma = data_tools.moving_average(COMvelocity[i, cut_beginning:], N)
        # moving_avg_line, = plt.plot(t_ma, ma, color="black", linewidth="1.5")

        individ_line, = plt.plot(t, COMvelocity[i, cut_beginning:], color="black", linewidth="1.5")

        # showing collision times
        # if summary['num_runs'] > 1:
        # for rob_i in range(summary['num_robots']):
        #     mask = [np.where(t==elem)[0][0] for elem in collision_times[i][rob_i] / 1000]
        #     plt.scatter(t[mask], COMvelocity[i, mask], color="red")

        for rob_i in range(summary['num_robots']):
            types = [elem[1] for elem in col_types[i][rob_i]]
            colors = ['red' if type == "withRobot" else 'gray' for type in types]
            col_times = [elem[0] / 1000 for elem in col_types[i][rob_i]]
            mask = [np.where(t == elem)[0][0] for elem in col_times]
            plt.scatter(t[mask], COMvelocity[i, mask], color=colors)

    if show_plot:
        plt.show()

def plot_reflection_effect_meanIID(summary, data, ax=None):
    """Showing the effect of being reflected from walls in a single experiment with multiple runs"""
    if ax is None:
        show_plot = True
        fig, ax = plt.subplots(1, summary['num_runs'], figsize=[10, 8])
    else:
        show_plot = False

    iid = data_tools.calculate_interindividual_distance(summary, data)
    population_mean = np.mean(np.mean(iid, 1), 1)
    col_types = data_tools.get_collisions_with_type(summary, data)
    t = data[0, 0, 0, :]

    for i in range(summary['num_runs']):

        if summary['num_runs'] > 1:
            plt.axes(ax[i])
        else:
            plt.axes(ax)

        # plotting ploarization over runs
        N = 300
        t_ma = t[int(N/2)-1:int(-N/2)]
        ma = data_tools.moving_average(population_mean[i, :], N)
        # moving_avg_line, = plt.plot(t_ma, ma, color="black", linewidth="1.5")

        individ_line, = plt.plot(t, population_mean[i, :], color="black", linewidth="1.5")

        # showing collision times
        # if summary['num_runs'] > 1:
        for rob_i in range(summary['num_robots']):
            types = [elem[1] for elem in col_types[i][rob_i]]
            colors = ['red' if type=="withRobot" else 'gray' for type in types]
            col_times = [elem[0]/1000 for elem in col_types[i][rob_i]]
            mask = [np.where(t==elem)[0][0] for elem in col_times]
            plt.scatter(t[mask], population_mean[i, mask], color=colors)

    if show_plot:
        plt.show()


def plot_reflection_effect_summary(summary, data):
    fig, ax = plt.subplots(3, summary['num_runs'], figsize=[10, 8], sharex='col', sharey='row')

    if summary['num_runs'] > 1:
        plot_reflection_effect_COMvelocity(summary, data, ax[0, :])
        plot_reflection_effect_polarization(summary, data, ax[1, :])
        plot_reflection_effect_meanIID(summary, data, ax[2, :])
        plt.axes(ax[0, 0])
        plt.ylabel("COM velocity [m/s]")
        plt.axes(ax[1, 0])
        plt.ylabel("Polarization [%]")
        plt.axes(ax[2, 0])
        plt.ylabel("Mean i-i.d. [m]")
        for i in range(summary['num_runs']):
            plt.axes(ax[2, i])
            plt.xlabel("time [s]")
    else:
        plot_reflection_effect_COMvelocity(summary, data, ax[0])
        plot_reflection_effect_polarization(summary, data, ax[1])
        plot_reflection_effect_meanIID(summary, data, ax[2])
        plt.axes(ax[0])
        plt.ylabel("COM velocity [m/s]")
        plt.axes(ax[1])
        plt.ylabel("Polarization [%]")
        plt.axes(ax[2])
        plt.ylabel("Mean i-i.d. [m]")
        plt.xlabel("time [s]")

    plt.subplots_adjust(wspace=0.15, hspace=0.05)


    plt.show()