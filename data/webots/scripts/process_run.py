import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def load_VSWRM_data(path):
    array2return = []
    with open(path, 'rb') as f:
        while True:
            try:
                array2return.append(pickle.load(f))
            except EOFError:
                break
    return np.array(array2return)


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


def plot_run_position(data_folder, robot_name, run_numbers, legends=None):
    """Plotting equilibrium distance according to an experimental data dict defined above"""
    if not isinstance(run_numbers, list):
        run_numbers = [run_numbers]

    if legends is not None:
        if not isinstance(legends, list):
            legends = [legends]

    fig, ax = plt.subplots(2, 2, figsize=[10, 8])

    for run_number in run_numbers:
        position_array = np.load(
            os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_pos.npy'))

        fake_time = np.linspace(0, 1, len(position_array))

        plt.axes(ax[0, 0])
        plt.scatter(position_array[:, 0], position_array[:, 2])

        plt.axes(ax[1, 0])
        plt.plot(fake_time, position_array[:, 0])

        plt.axes(ax[1, 1])
        plt.plot(fake_time, position_array[:, 2])

    plt.axes(ax[0, 0])
    plt.title(f'Trajectory of {robot_name}')
    plt.xlabel('timestep [n]')
    plt.ylabel('position [x, y]')
    if legends is not None:
        plt.legend(legends)

    plt.axes(ax[1, 0])
    plt.title(f'{robot_name} x coordinate')
    plt.xlabel('timestep [n]')
    plt.ylabel('position [x]')
    if legends is not None:
        plt.legend(legends)

    plt.axes(ax[1, 1])
    plt.title(f'{robot_name} y coordinate')
    plt.xlabel('timestep [n]')
    plt.ylabel('position [y]')
    if legends is not None:
        plt.legend(legends)

    plt.show()


def plot_run_position_pendulum(data_folder, robot_name, run_numbers, legends=None, suptitle=None, simulation_time=None,
                               with_orientation=False):
    """Plotting equilibrium distance according to an experimental data dict defined above"""
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
        position_array = np.load(
            os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_pos.npy'))

        fake_time = np.linspace(0, 1, len(position_array))
        if simulation_time is not None:
            fake_time = fake_time * simulation_time

        plt.axes(ax[0])
        plt.plot(fake_time, position_array[:, 0])

        plt.axes(ax[1])
        plt.plot(fake_time, position_array[:, 2])

        if with_orientation:
            or_array = np.load(
                os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_or.npy'))
            plt.axes(ax[2])
            plt.plot(fake_time, or_array)

    if suptitle is None:
        plt.suptitle('Pendulum movement for different simulation timesteps')
    else:
        plt.suptitle(suptitle)

    plt.axes(ax[0])
    plt.title(f'{robot_name} x coordinate')
    if simulation_time is not None:
        plt.xlabel('simulation time [V-sec]')
    else:
        plt.xlabel('simulation time [AU]')
    plt.ylabel('position [m]')
    if legends is not None:
        plt.legend(legends)

    plt.axes(ax[1])
    plt.title(f'{robot_name} y coordinate')
    if simulation_time is not None:
        plt.xlabel('simulation time [V-sec]')
    else:
        plt.xlabel('simulation time [AU]')
    plt.ylabel('position [m]')
    if legends is not None:
        plt.legend(legends)

    plt.axes(ax[2])
    plt.title(f'{robot_name} orientation')
    if simulation_time is not None:
        plt.xlabel('simulation time [V-sec]')
    else:
        plt.xlabel('simulation time [AU]')
    plt.ylabel('orientation [rad]')
    if legends is not None:
        plt.legend(legends)

    plt.show()


def plot_position(data_folder, robot_name, run_numbers, legends=None, suptitle=None, with_orientation=False):
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
        position_array = load_VSWRM_data(
            os.path.join(data_folder, robot_name, run_number, f'{robot_name}_run{run_number}_pos.npy'))

        fake_time = position_array[:, 0] / 1000

        plt.axes(ax[0])
        plt.plot(fake_time, position_array[:, 1])

        plt.axes(ax[1])
        plt.plot(fake_time, position_array[:, 3])

        if with_orientation:
            or_array = load_VSWRM_data(
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

    plt.show()
