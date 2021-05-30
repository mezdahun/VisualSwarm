"""
@author: mezdahun
@description: tools AFTER WeBots simulation and processing to transform, read, process, summarize, etc. collected data
"""
import os
import pickle
import json
import numpy as np


def load_VSWRM_data(path):
    array2return = []
    with open(path, 'rb') as f:
        while True:
            try:
                array2return.append(pickle.load(f))
            except EOFError:
                break
    return np.array(array2return)


def summarize_experiment(data_path, experiment_name):
    """This method summarizes separated WeBots simulation data into a unified satastructure. All measurements must
    have the same length. To accomplish this you should use a positive non-zero PAUSE_SIMULATION_AFTER parameter
    passed from webots.  All robot folders should contain the same number of run folders. Mixed folders resulting from
    changing the number of robots throughout the runs of the experiments will not be handled.

    The output will be saved under data_path in 2 separate files, 1 holding numeric data *_summaryd.npy the other
    the corresponding metadata as a dictinary *_summaryp.json

    The numeric data has the shape of

            (number of runs) x (number of robots) x (number of saved attributes) x (simulation timesteps)

    and the first attribute is always the simulation time"""

    attributes = ['t', 'pos_x', 'pos_y', 'pos_z', 'or']
    num_attributes = len(attributes)

    robots = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    num_robots = len(robots)

    param_dict = {}

    for i, robot_name in enumerate(robots):
        robot_folder = os.path.join(data_path, robot_name)

        if i == 0:
            runs = [name for name in os.listdir(robot_folder) if os.path.isdir(os.path.join(robot_folder, name))]
            num_runs = len(runs)

        for j, run_name in enumerate(runs):

            position_array = load_VSWRM_data(
                os.path.join(data_path, robot_name, run_name, f'{robot_name}_run{run_name}_pos.npy'))

            or_array = load_VSWRM_data(
                os.path.join(data_path, robot_name, run_name, f'{robot_name}_run{run_name}_or.npy'))

            with open(os.path.join(data_path, robot_name, run_name,
                                   f'{robot_name}_run{run_name}_params.json')) as param_f:
                param_dict[f'run{run_name}'] = json.load(param_f)

            if i == 0 and j == 0:
                t = position_array[:, 0] / 1000
                t_len = len(t)
                data = np.zeros((num_runs, num_robots, num_attributes, t_len))

            data[j, i, attributes.index('t'), :] = t
            data[j, i, attributes.index('pos_x'), :] = position_array[:, 1]
            data[j, i, attributes.index('pos_y'), :] = position_array[:, 2]
            data[j, i, attributes.index('pos_z'), :] = position_array[:, 3]
            data[j, i, attributes.index('or'), :] = or_array[:, 1]

    experiment_summary = {'params': param_dict,
                          'num_runs': num_runs,
                          'num_robots': num_robots,
                          'num_attributes': num_attributes,
                          'attributes': attributes,
                          'experiment_name': experiment_name}

    with open(os.path.join(data_path, f'{experiment_name}_summaryp.json'), 'w') as sump_f:
        json.dump(experiment_summary, sump_f, indent=4)

    sumd_f = os.path.join(data_path, f'{experiment_name}_summaryd.npy')
    np.save(sumd_f, data)


def read_summary_data(data_path, experiment_name):
    with open(os.path.join(data_path, f'{experiment_name}_summaryp.json')) as sump_f:
        summary_dict = json.load(sump_f)

    sumd_f = os.path.join(data_path, f'{experiment_name}_summaryd.npy')
    data = np.load(sumd_f)

    return summary_dict, data


def velocity(position_array, orientation_array, time):
    """position_array shall be of shape (3xN) where N is the number of
    datapoints across time. orientation_array and time shall be N length 1D array"""
    velocities = []

    for t in range(1, position_array.shape[1]):
        if -np.pi / 2 < orientation_array[t] < np.pi / 2:  # aligned with positive z direction
            or_sign = np.sign(position_array[2, t] - position_array[2, t - 1])
        else:
            or_sign = - np.sign(position_array[2, t] - position_array[2, t - 1])
        velocities.append(or_sign * distance(position_array[:, t], position_array[:, t - 1]) / (time[t] - time[t - 1]))

    return np.array([velocities])


def distance(p1, p2):
    """distance of 3D points. Multiple points can be passed in a (3xN) shape
    numpy array where N is the number of points.

    returns an N length 1D array holding the pairwise distances of
    the passed points"""
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


def calculate_velocity(summary, data):
    velocities = np.zeros((summary['num_runs'], summary['num_robots'], data.shape[-1] - 1))

    t_idx = summary['attributes'].index('t')
    pos_x = summary['attributes'].index('pos_x')
    pos_y = summary['attributes'].index('pos_y')
    pos_z = summary['attributes'].index('pos_z')
    or_idx = summary['attributes'].index('or')

    for i in range(summary['num_runs']):
        for j in range(summary['num_robots']):
            velocities[i, j, :] = velocity(data[i, j, [pos_x, pos_y, pos_z], :],
                                           data[i, j, or_idx, :],
                                           data[i, j, t_idx, :])

    return velocities


def calculate_distance(summary, data, from_point):
    """Calculating robot ditances from a given point"""

    distances = np.zeros((summary['num_runs'], summary['num_robots'], data.shape[-1]))

    t_idx = summary['attributes'].index('t')
    pos_x = summary['attributes'].index('pos_x')
    pos_y = summary['attributes'].index('pos_y')
    pos_z = summary['attributes'].index('pos_z')

    for i in range(summary['num_runs']):
        for j in range(summary['num_robots']):
            pos_array = data[i, j, [pos_x, pos_y, pos_z], :]
            if i == 0 and j == 0:
                reference = np.zeros_like(pos_array)
                reference[0, :] = from_point[0]
                reference[1, :] = from_point[1]
                reference[2, :] = from_point[2]

            distances[i, j, :] = distance(pos_array, reference)

    return distances


def calculate_interindividual_distance(summary, data):
    iid = np.zeros((summary['num_runs'], summary['num_robots'], summary['num_robots'], data.shape[-1]))

    t_idx = summary['attributes'].index('t')
    pos_x = summary['attributes'].index('pos_x')
    pos_y = summary['attributes'].index('pos_y')
    pos_z = summary['attributes'].index('pos_z')

    for runi in range(summary['num_runs']):
        for robi in range(summary['num_robots']):
            for robj in range(summary['num_robots']):
                pos_array_i = data[runi, robi, [pos_x, pos_y, pos_z], :]
                pos_array_j = data[runi, robj, [pos_x, pos_y, pos_z], :]
                iid[runi, robi, robj, :] = distance(pos_array_i, pos_array_j)

    return iid


def calculate_mean_iid(summary, data, window_width=100):
    iid = calculate_interindividual_distance(summary, data)
    for i in range(summary['num_robots']):
        iid[:, i, i, :] = np.inf

    miid = np.zeros((summary['num_runs'],))

    for i in range(summary['num_runs']):
        miid[i] = np.mean(np.min(np.min(iid[i, :, :, -window_width::], axis=0), axis=0))

    return miid


def calculate_ploarization_matrix(summary, data):

    time = data[0, 0, 0, :]  # 1 step shorter because of diff

    or_idx = summary['attributes'].index('or')
    orientations = data[:, :, or_idx, :]

    pol = np.zeros((summary['num_runs'], summary['num_robots'], summary['num_robots'],data.shape[-1]))

    for i in range(summary['num_robots']):
        pol[:, i, i, :] = np.nan

    for i in range(summary['num_runs']):
        for ri in range(summary['num_robots']):
            for rj in range(summary['num_robots']):
                diff = np.abs(orientations[i, ri, :] - orientations[i, rj, :])
                pol[i, ri, rj, :] = ((2 / np.pi) * np.abs(diff - np.pi)) - 1

    return pol


def calculate_mean_polarization(summary, data, window_width=100):

    pol = calculate_ploarization_matrix(summary, data)
    mean_pol = np.zeros(summary['num_runs'])

    for i in range(summary['num_runs']):
        p_vec = np.zeros(window_width)
        norm_fac = 0
        for ri in range(summary['num_robots']):
            for rj in range(ri, summary['num_robots']):
                p_vec += pol[i, ri, rj, -window_width::]
                norm_fac += 1
        print(norm_fac)
        mean_pol[i] = np.mean((p_vec/(norm_fac)))

    return mean_pol


def calculate_min_iid(summary, data):
    iid = calculate_interindividual_distance(summary, data)
    for i in range(summary['num_robots']):
        iid[:, i, i, :] = np.inf

    min_iid = np.zeros((summary['num_runs'],))

    for i in range(summary['num_runs']):
        min_iid[i] = np.min(iid[i, :, :, :])

    return min_iid


def population_velocity(summary, data):
    """Calculating the velocity and direction of velocity of the center of mass of agents

    returns an (n_runs x (t-1)) shape matrix"""
    pos_x = summary['attributes'].index('pos_x')
    pos_y = summary['attributes'].index('pos_y')
    pos_z = summary['attributes'].index('pos_z')
    center_of_mass = np.mean(data[:, :, [pos_x, pos_y, pos_z], :], axis=1)

    t = data[0, 0, 0, :]
    dt = t[1] - t[0]

    COMvelocity = np.zeros((summary['num_runs'], len(t)-1))
    for i in range(len(t)-1):
        for run_i in range(summary['num_runs']):
            COMvelocity[run_i, i] = distance(center_of_mass[run_i, :, i], center_of_mass[run_i, :, i+1]) / dt

    return COMvelocity
