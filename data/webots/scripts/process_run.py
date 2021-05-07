import numpy as np
import matplotlib.pyplot as plt
import os


def plot_run_orientation(data_folder, robot_name, run_number):
    """Plotting equilibrium distance according to an experimental data dict defined above"""
    orientation_array = np.load(
        os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_or.npy'))
    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    plt.plot(orientation_array, color='green', marker='.', linewidth=0, markersize=16)
    plt.title(f'Orientation of {robot_name}')
    plt.xlabel('timestep [n]')
    plt.ylabel('orientation [rad]')
    plt.show()


def plot_run_position(data_folder, robot_name, run_number):
    """Plotting equilibrium distance according to an experimental data dict defined above"""
    position_array = np.load(
        os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_pos.npy'))
    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    plt.scatter(position_array[:, 0], position_array[:, 2])
    plt.title(f'Position of {robot_name}')
    plt.xlabel('timestep [n]')
    plt.ylabel('position [x, y]')
    plt.show()
