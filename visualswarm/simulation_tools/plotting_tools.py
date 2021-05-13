"""
@author: mezdahun
@description: tools to plot recorded values in webots (only works with saving data)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from visualswarm.simulation_tools import data_tools
import pickle
import json


def plot_position(data_folder, robot_name, run_numbers, legends=None, suptitle=None, with_orientation=False,
                  with_save=False):
    """Plotting position and orientation of a single robot possibly for multiple runs"""
    if not isinstance(run_numbers, list):
        run_numbers = [run_numbers]

    if legends is not None:
        if not isinstance(legends, list):
            legends = [legends]

    if not with_orientation:
        fig, ax = plt.subplots(2, 1, figsize=[10, 8])
    else:
        fig, ax = plt.subplots(3, 1, figsize=[10, 8])

    for run_number in run_numbers:
        position_array = data_tools.load_VSWRM_data(
            os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_pos.npy'))

        fake_time = position_array[:, 0] / 1000

        plt.axes(ax[0])
        plt.plot(fake_time, position_array[:, 1])

        plt.axes(ax[1])
        plt.plot(fake_time, position_array[:, 3])

        if with_orientation:
            or_array = data_tools.load_VSWRM_data(
                os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_or.npy'))
            plt.axes(ax[2])
            plt.plot(fake_time, or_array[:, 1])

    if suptitle is None:
        plt.suptitle(f'Position of robot {robot_name}')
    else:
        plt.suptitle(suptitle)

    plt.axes(ax[0])
    plt.title(f'{robot_name} x coordinate')
    plt.xlabel('simulation time [V-sec]')
    plt.ylabel('position [m]')
    if legends is not None:
        plt.legend(legends)

    plt.axes(ax[1])
    plt.title(f'{robot_name} y coordinate')
    plt.xlabel('simulation time [V-sec]')
    plt.ylabel('position [m]')
    if legends is not None:
        plt.legend(legends)

    if with_orientation:
        plt.axes(ax[2])
        plt.title(f'{robot_name} orientation')
        plt.xlabel('simulation time [V-sec]')
        plt.ylabel('orientation [rad]')
        if legends is not None:
            plt.legend(legends)

    if with_save:
        figpath = os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_plot.jpg')
        plt.savefig(figpath)
        plt.close()
    else:
        plt.show()


def calculate_distance(p1, p2):
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


def plot_distance(data_folder, robot_names, run_number, legends=None, suptitle=None, with_orientation=False,
                  with_save=False, from_fixed_pos=None):
    if isinstance(run_number, list):
        raise Exception('More than a single run number is passed, using the first one')
        run_number = run_number[0]

    num_robots = len(robot_names)

    if num_robots > 2:
        raise Exception('Plotting method implemented for 2 robots only')

    robot_positions = []

    for robot_name in robot_names:
        position_array = data_tools.load_VSWRM_data(
            os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_pos.npy'))
        robot_positions.append(position_array)

    if from_fixed_pos is None:
        # distance between moving robots

        distance_array = np.zeros(np.min((len(robot_positions[0][:, 0]), len(robot_positions[1][:, 0]))))
        fake_time = np.zeros(np.min((len(robot_positions[0][:, 0]), len(robot_positions[1][:, 0]))))

        for t in range(len(distance_array)):
            distance = calculate_distance(robot_positions[0][t, 1::], robot_positions[1][t, 1::])
            distance_array[t] = distance
            fake_time[t] = robot_positions[0][t, 0]

        fig, ax = plt.subplots(1, 1, figsize=[10, 8])
        plt.plot(fake_time, distance)

    else:

        distance_array = []
        fake_time = []

        robot_position = robot_positions[0]
        for t in range(len(robot_position)):
            distance = calculate_distance(robot_position[t, 1::], from_fixed_pos)
            distance_array.append(distance)
            fake_time.append(robot_position[t, 0])

        fig, ax = plt.subplots(1, 1, figsize=[10, 8])
        plt.plot(distance_array)

    plt.show()


def plot_velocities(summary, data, changed_along=None, changed_along_alias=None):
    velocities = data_tools.calculate_velocity(summary, data)
    fig, ax = plt.subplots(summary['num_robots'], 1, figsize=[10, 8])

    time = data[0, 0, 0, :-1]  # 1 step shorter because of diff

    for i in range(summary['num_robots']):
        if summary['num_robots'] > 1:
            plt.axes(ax[i])

        plt.plot(time, velocities[:, i, :].T)

        if changed_along is not None:
            if changed_along_alias is None:
                legend = [f'{changed_along}={summary["params"][f"run{j+1}"][changed_along]}' for j in range(summary['num_runs'])]
            else:
                legend = [f'{changed_along_alias}={summary["params"][f"run{j + 1}"][changed_along]}' for j in
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

    for i in range(summary['num_robots']):
        if summary['num_robots'] > 1:
            plt.axes(ax[i])
        plt.plot(time, distances[:, i, :].T)

        if changed_along is not None:
            if changed_along_alias is None:
                legend = [f'{changed_along}={summary["params"][f"run{j+1}"][changed_along]}' for j in range(summary['num_runs'])]
            else:
                legend = [f'{changed_along_alias}={summary["params"][f"run{j + 1}"][changed_along]}' for j in
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

    for i in range(summary['num_robots']):
        if summary['num_robots'] > 1:
            plt.axes(ax[i])
        plt.plot(time, orientations[:, i, :].T)

        if changed_along is not None:
            if changed_along_alias is None:
                legend = [f'{changed_along}={summary["params"][f"run{j+1}"][changed_along]}' for j in range(summary['num_runs'])]
            else:
                legend = [f'{changed_along_alias}={summary["params"][f"run{j + 1}"][changed_along]}' for j in
                          range(summary['num_runs'])]
            plt.legend(legend)

        plt.title(f'Orientation of robot {i}')
        plt.xlabel('simulation time [s]')
        plt.ylabel('orientation [rad]')

    plt.show()