"""
@author: mezdahun
@description: tools to plot recorded values in webots (only works with saving data)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from visualswarm.simulation_tools import data_tools


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