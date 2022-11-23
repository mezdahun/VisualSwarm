"""
@author: mezdahun
@description: tools AFTER WeBots simulation and processing to transform, read, process, summarize, etc. collected data
"""
import os
import pickle  # nosec
import json
import numpy as np
import logging
import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def load_VSWRM_data(path):
    array2return = []
    with open(path, 'rb') as f:
        while True:
            try:
                # TODO: check if data was generated by VSWRM and refuse to deserialize otherwise
                array2return.append(pickle.load(f))  # nosec
            except EOFError:
                break
    return np.array(array2return)


def find_min_t(data_path):
    """as both different runs and different robots can yield slightly different number of measurement points
    we should first find the minimum so that all others are limited to this minimum"""
    robots = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    min_ts = []
    for i, robot_name in enumerate(robots):
        robot_folder = os.path.join(data_path, robot_name)

        if i == 0:
            runs = [name for name in os.listdir(robot_folder) if os.path.isdir(os.path.join(robot_folder, name))]
            num_runs = len(runs)

        t_lens = []
        for j, run_name in enumerate(runs):
            position_array = load_VSWRM_data(
                os.path.join(data_path, robot_name, run_name, f'{robot_name}_run{run_name}_pos.npy'))

            t_lens.append(len(position_array[:, 0]))

        min_ts.append(np.min(t_lens))

    return np.min(min_ts)


def is_summarized(data_path, experiment_name):
    """Checking if a given experiment is already summarized into data and summary files so we can skip the rather
    long summary time for already summed experiments"""
    is_summary = os.path.isfile(os.path.join(data_path, f'{experiment_name}_summaryp.json'))
    is_data = os.path.isfile(os.path.join(data_path, f'{experiment_name}_summaryd.npy'))
    if is_summary and is_data:
        logger.info(f"{experiment_name} is already summarized...")
        return True
    else:
        logger.info("Experiment not yet summarized")
        return False

def optitrackcsv_to_VSWRM(csv_path, skip_already_summed=True):
    """Reading an exported optitrack tracking data csv file into a VSWRM summary sata file that can be further used
    for data analysis and plotting with VSWRM

    The exported file must contain 6 columns for each robot, that are
    rotationX, rotationY, rotationZ, positionX, positionY, positionZ

    timepoints where tracking of robots has been lost will be simply cut from the data."""

    data_path = os.path.dirname(csv_path)
    _, csv_filename = os.path.split(csv_path)
    experiment_name = csv_filename.split('.')[0]
    if is_summarized(data_path, experiment_name) and skip_already_summed:
        logger.info(f"Skipping already summed experiment as requested!")
        return True

    df_orig = pd.read_csv(csv_path, skiprows=[i for i in range(6)])

    # QUICKNDIRTY dropping timepoints where optitrack lost track of robots
    df = df_orig.dropna()
    print("Dropped NA values from dataframe.")
    print("columns: ", df.columns)

    num_robots = int(len(df.columns) / 6) # for each robot 3 rotation and 3 position coordinate
    print(f"Found {num_robots} robots data in csv file.")
    time = df['Time (Seconds)'].values
    print(time)
    # time = np.array([a[1] for a in df.index.values[5::]]).astype('float') / 1000
    t_len = len(time)

    attributes = ['t', 'pos_x', 'pos_y', 'pos_z', 'or']
    num_attributes = len(attributes)

    data = np.zeros((1, num_robots, num_attributes, t_len))
    from scipy.spatial.transform import Rotation

    for robi in range(num_robots):
        startindex = int(robi * 6) + 2
        orient_x = df.iloc[:, startindex + 0].values.astype('float') #x axis
        orient_y = df.iloc[:, startindex + 1].values.astype('float') #y axis
        orient_z = df.iloc[:, startindex + 2].values.astype('float') #z axis
        orient = np.array([Rotation.from_euler('xyz', [orient_x[i], orient_y[i], orient_z[i]], degrees=True).as_euler('yxz', degrees=False)[0] for i in range(len(orient_y))])
        orient = - np.pi/2 - (orient + np.pi)
        x_pos = df.iloc[:, startindex + 3].values.astype('float')
        y_pos = df.iloc[:, startindex + 4].values.astype('float')
        z_pos = df.iloc[:, startindex + 5].values.astype('float')

        data[0, robi, attributes.index('t'), :] = time
        data[0, robi, attributes.index('pos_x'), :] = x_pos
        data[0, robi, attributes.index('pos_y'), :] = y_pos
        data[0, robi, attributes.index('pos_z'), :] = z_pos
        data[0, robi, attributes.index('or'), :] = orient


    experiment_summary = {'params': None,
                          'num_runs': 1,
                          'num_robots': num_robots,
                          'num_attributes': num_attributes,
                          'attributes': attributes,
                          'experiment_name': experiment_name}


    with open(os.path.join(data_path, f'{experiment_name}_summaryp.json'), 'w') as sump_f:
        json.dump(experiment_summary, sump_f, indent=4)

    sumd_f = os.path.join(data_path, f'{experiment_name}_summaryd.npy')
    np.save(sumd_f, data)






def summarize_experiment(data_path, experiment_name, skip_already_summed=True):
    """This method summarizes separated WeBots simulation data into a unified satastructure. All measurements must
    have the same length. To accomplish this you should use a positive non-zero PAUSE_SIMULATION_AFTER parameter
    passed from webots.  All robot folders should contain the same number of run folders. Mixed folders resulting from
    changing the number of robots throughout the runs of the experiments will not be handled.

    The output will be saved under data_path in 2 separate files, 1 holding numeric data *_summaryd.npy the other
    the corresponding metadata as a dictinary *_summaryp.json

    The numeric data has the shape of

            (number of runs) x (number of robots) x (number of saved attributes) x (simulation timesteps)

    and the first attribute is always the simulation time"""

    if is_summarized(data_path, experiment_name) and skip_already_summed:
        logger.info(f"Skipping already summed experiment as requested!")
        return True

    attributes = ['t', 'pos_x', 'pos_y', 'pos_z', 'or']
    num_attributes = len(attributes)

    robots = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    num_robots = len(robots)

    param_dict = {}

    t_len = find_min_t(data_path)

    for i, robot_name in enumerate(robots):
        robot_folder = os.path.join(data_path, robot_name)

        if i == 0:
            runs = [name for name in os.listdir(robot_folder) if os.path.isdir(os.path.join(robot_folder, name))]
            num_runs = len(runs)

        positions = []
        orientations = []
        # t_lens = []
        for j, run_name in enumerate(runs):

            position_array = load_VSWRM_data(
                os.path.join(data_path, robot_name, run_name, f'{robot_name}_run{run_name}_pos.npy'))
            positions.append(position_array)

            or_array = load_VSWRM_data(
                os.path.join(data_path, robot_name, run_name, f'{robot_name}_run{run_name}_or.npy'))
            orientations.append(or_array)

            # t_lens.append(len(position_array[:, 0]))

            with open(os.path.join(data_path, robot_name, run_name,
                                   f'{robot_name}_run{run_name}_params.json')) as param_f:
                if i == 0:
                    param_dict[f'run{run_name}'] = json.load(param_f)
                er_times_path = os.path.join(data_path, robot_name, run_name,
                                             f'{robot_name}_run{run_name}_ERtimes.json')
                if os.path.isfile(er_times_path):
                    with open(er_times_path) as erf:
                        if param_dict[f'run{run_name}'].get('ERtimes') is None:
                            param_dict[f'run{run_name}']['ERtimes'] = {}
                        param_dict[f'run{run_name}']['ERtimes'][robot_name] = json.load(erf)

        if i == 0:
            # t_len = np.min(t_lens)
            t = positions[0][:t_len, 0] / 1000
            data = np.zeros((num_runs, num_robots, num_attributes, t_len))

        for j, run_name in enumerate(runs):
            data[j, i, attributes.index('t'), :] = t
            data[j, i, attributes.index('pos_x'), :] = positions[j][:t_len, 1]
            data[j, i, attributes.index('pos_y'), :] = positions[j][:t_len, 2]
            data[j, i, attributes.index('pos_z'), :] = positions[j][:t_len, 3]
            data[j, i, attributes.index('or'), :] = orientations[j][:t_len, 1]

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
    """Calculating matrix including polariuzation values between 0 and 1 reflecting the
    average match of the agents w.r.t. heaing angles"""
    time = data[0, 0, 0, :]  # 1 step shorter because of diff

    or_idx = summary['attributes'].index('or')
    orientations = data[:, :, or_idx, :]

    pol = np.zeros((summary['num_runs'], summary['num_robots'], summary['num_robots'], data.shape[-1]))

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
        mean_pol[i] = np.mean((p_vec / (norm_fac)))

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

    COMvelocity = np.zeros((summary['num_runs'], len(t) - 1))
    for i in range(len(t) - 1):
        for run_i in range(summary['num_runs']):
            COMvelocity[run_i, i] = distance(center_of_mass[run_i, :, i], center_of_mass[run_i, :, i + 1]) / dt

    return COMvelocity


def get_collision_time_intervals(summary):
    """Calculating larger collision time intervals for all robot and run according to raw recorded ERtimes
    in summary data"""

    collision_intervals = dict.fromkeys(range(0, summary['num_runs']),
                                        dict.fromkeys(range(0, summary['num_robots']), []))
    collision_times = {}
    for i in range(summary['num_runs']):
        collision_times[i] = {}
        for j in range(summary['num_robots']):
            r_sum = summary['params'][f'run{i + 1}']
            col_times_dict = r_sum.get("ERtimes")
            if col_times_dict is not None:
                rob_col_times = col_times_dict.get(f'robot{j}')
                if rob_col_times is not None:
                    collision_times[i][j] = np.array(rob_col_times.get("ERtimes"))

    # from pprint import pprint
    # r_i = 0
    # for run_name, r_sum in summary['params'].items():
    #     col_times_dict = r_sum.get("ERtimes")  # all collisions in run
    #
    #     if col_times_dict is not None:  # there were collisions during the experiment, looping through robots
    #         for robi in range(summary['num_robots']):
    #             rob_col_times = col_times_dict.get(f'robot{robi}')
    #             if rob_col_times is not None:
    #                 ctimes = np.array(rob_col_times.get("ERtimes"))
    #                 # print(ctimes / 1000)
    #                 collision_times[r_i][robi] = ctimes.copy()
    #                 # print(collision_times[r_i][robi])
    #
    #                 # time difference between 2 ER reports is larger than reporting frequency
    #                 mask_temp = list(np.where(np.diff(ctimes) > 300)[0])
    #                 mask = [0]  # first element always border
    #                 mask.append(len(ctimes) - 1)  # last element always border
    #                 mask.extend(mask_temp)
    #
    #                 mask_temp = list(np.where(np.diff(ctimes) > 300)[0] + 1)  # get end of intervals
    #                 mask.extend(mask_temp)
    #                 collision_intervals[r_i][robi] = np.array(sorted(list(ctimes[mask])))
    #
    #     r_i += 1

    return collision_intervals, collision_times


def get_collisions_with_type(summary, data, range_around_col=0.3, BL_thr=0.18):
    """Returns the collision data from the experiments including the type of the collisions, that can be
    robot-robot vs robot-static. The former one is defined according to any robot-robot distance did go below
    BL_thr within <range_around_col> timerange (in sec) from the timepoint of collision"""
    t = data[0, 0, 0, :]
    dt = t[1] - t[0]
    range_steps = int(range_around_col / dt)

    coll_intervals, coll_times = get_collision_time_intervals(summary)

    iid_matrix = calculate_interindividual_distance(summary, data)
    for i in range(summary['num_runs']):
        for j in range(summary['num_robots']):
            iid_matrix[i, j, j, :] = np.inf

    collision_times = dict.fromkeys(range(0, summary['num_runs']),
                                    dict.fromkeys(range(0, summary['num_robots']), []))

    for i in range(summary['num_runs']):
        if i in coll_times.keys():
            for j in range(summary['num_robots']):
                if j in coll_times[i].keys():
                    for ct in sorted(coll_times[i][j]):
                        t_start, = np.where(np.isclose(t, ct / 1000))
                        if len(t_start) == 0:
                            print(i, j, 'WARNING: skip step with ', ct / 1000)
                            continue
                        t_start = int(t_start)
                        t_end = np.min([t_start + range_steps, len(t) - 1])
                        t_end = int(t_end)
                        if np.min(iid_matrix[i, j, :, t_start:t_end]) < BL_thr:
                            # print('COLLISION WITH ROBOT')
                            # print(f"run{i}@{ct / 1000}")
                            # print(np.min(iid_matrix[i, j, :, t_start:t_end]))
                            collision_times[i][j].append((ct, 'withRobot'))
                        else:
                            collision_times[i][j].append((ct, 'withStatic'))

    return collision_times


def get_robot_collision_ratio(summary, data):
    """Returns the ratio of robot-robot vs robot-static collisions throughout the whole experiment with all runs"""
    collisions = get_collisions_with_type(summary, data)
    with_robot = []
    with_static = []
    for i in range(summary['num_runs']):
        for j in range(summary['num_robots']):
            with_robot.extend([elem[0] for elem in collisions[i][j] if elem[1] == 'withRobot'])
            with_static.extend([elem[0] for elem in collisions[i][j] if elem[1] == 'withStatic'])
    if (len(with_static) + len(with_robot)) > 0:
        return len(with_robot) / (len(with_static) + len(with_robot)), len(with_robot), len(with_static)
    else:
        return 0, len(with_robot), len(with_static)


def moving_average(x, N, weights=1):
    """Simple moving average with window length N"""
    return np.convolve(x, np.ones(N) * weights, 'valid') / N


def is_crystallized(summary, data, num_run, time_window=5, vel_thr=0.003, polvar_thr=0.3, pol_thr=0.5):
    """Returns a bool showing if a given run in an experiment was stuck in a crystallized state according to the
    population velocity and the variance of the polarization of agents in the last <time_window> seconds"""

    t = data[0, 0, 0, :-1]
    dt = t[1] - t[0]
    num_timesteps = int(time_window / dt)

    COM_vel = population_velocity(summary, data)[num_run, :]
    polarization = calculate_ploarization_matrix(summary, data)
    population_mean = np.mean(np.mean(polarization, 1), 1)[num_run, :]

    # print('mean COM velocity: ', np.mean(COM_vel[-num_timesteps:]))
    # print('std polarization: ', np.std(population_mean[-num_timesteps:]))
    # print('men polarization: ', np.mean(population_mean[-num_timesteps:]))
    if np.mean(COM_vel[-num_timesteps:]) < vel_thr and np.std(population_mean[-num_timesteps:]) < polvar_thr \
            and np.mean(population_mean[-num_timesteps:]) < pol_thr:
        return True
    else:
        return False


def return_mean_polarization_at_end(summary, data, time_window=5):
    """Returns the mean polarization in the last <time_window> seconds of the experiment"""
    t = data[0, 0, 0, :-1]
    dt = t[1] - t[0]
    num_timesteps = int(time_window / dt)

    polarization = calculate_ploarization_matrix(summary, data)
    population_mean = np.mean(np.mean(np.mean(polarization, 1), 1), axis=0)[-num_timesteps:]
    return np.mean(population_mean)


def t_without_collision(summary, data, collision_removal_range=1):
    """Returning a time axis for each run from which timepoints are excluded
     where at least one of the agents collided"""

    t = data[0, 0, 0, :]
    dt = t[1] - t[0]
    num_timesteps = int(collision_removal_range / dt)

    ctimes = get_collisions_with_type(summary, data)
    filtered_ts = {}
    for i in range(summary['num_runs']):
        filtered_ts[i] = None

    for i in range(summary['num_runs']):
        if i in ctimes.keys():
            filtered_t = t.copy()
            for j in range(summary['num_robots']):
                if j in ctimes[i].keys():
                    for ct in ctimes[i][j]:
                        t_start, = np.where(np.isclose(t, ct[0] / 1000))
                        if len(t_start) == 0:
                            print(i, j, 'WARNING: skip step with ', ct[0] / 1000)
                            continue
                        t_start = int(t_start)
                        t_end = np.min([t_start + num_timesteps, len(t) - 1])
                        t_end = int(t_end)
                        filtered_t[t_start:t_end] = -1
            filtered_t2 = filtered_t.copy()
            filtered_t2 = filtered_t2[filtered_t2 > -1]
            filtered_ts[i] = filtered_t2
        else:
            filtered_ts[i] = t.copy()

    return filtered_ts


def return_mean_polarization_without_collision(summary, data, collision_removal_range=1):
    """Calculating the mean polarization of the group in those times when none of the robots collided"""
    noc_t = t_without_collision(summary, data, collision_removal_range=collision_removal_range)
    pol_matrix = calculate_ploarization_matrix(summary, data)

    t = data[0, 0, 0, :]

    run_means = []

    for i in range(summary['num_runs']):
        if i in noc_t.keys():
            _, _, x_ind = np.intersect1d(t, noc_t[i] / 1000, return_indices=True)
            polarization = pol_matrix[i, :, :, x_ind]
            run_means.append(np.mean(np.mean(np.mean(polarization, 1), 1)))

    return np.mean(run_means)


def time_spent_undesired_state(summary, data, vel_thr=0.02):
    """calculating time spent in an undesired state where the average COM velocity is zero of the group."""
    t = data[0, 0, 0, :-1]
    dt = t[1] - t[0]

    COM_vel = population_velocity(summary, data)

    times = []

    for i in range(summary['num_runs']):
        undes_state_mask = COM_vel[i, :] < vel_thr
        times.append(len(t[undes_state_mask]))

    return (np.mean(times) * dt) / t[-1]


def overall_COM_travelled_distance_while_polarized(summary, data, polarization_thr=0.7):
    """Calculating the overall travelled distance of the COM of the group while the group was polarized.
    This excludes those undesired states when the group is unpolarized but still pulling andpushing each other
    away and by that moving the COMN of the group."""
    t = data[0, 0, 0, :-1]
    dt = t[1] - t[0]

    polarization = calculate_ploarization_matrix(summary, data)
    mean_polarization = np.mean(np.mean(polarization, 1), 1)
    travelled_distances = []
    for i in range(summary['num_runs']):
        high_pol_mask = mean_polarization[i, :-1] > polarization_thr
        pop_distance = population_velocity(summary, data) * dt
        travelled_distances.append(np.sum(pop_distance[i, high_pol_mask]))

    return np.sum(travelled_distances)
