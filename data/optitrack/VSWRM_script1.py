"""
@description: cPOC for processing optitrack data for publication.
"""
"""EXPERIMENT DESCRIPTION:

@description: changing parameter gamma with a single controlled robot."""
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage

data_path = "/home/david/Desktop/database/OptiTrackCSVs"
EXPERIMENT_NAMES = [f"E8s1r{i+1}" for i in range(3)]
DISTANCE_REFERENCE = [0, 0, 0]


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


from visualswarm.simulation_tools import data_tools, plotting_tools

polrats_over_exps = []
for expi in range(len(EXPERIMENT_NAMES)):
    # if data is freshly created first summarize it into multidimensional array
    csv_path = os.path.join(data_path, f"{EXPERIMENT_NAMES[expi]}.csv")
    data_tools.optitrackcsv_to_VSWRM(csv_path)

    change_along = None
    change_along_alias = None

    # retreiving data
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAMES[expi])

    #
    # plotting_tools.plot_mean_COMvel_over_runs(summary, data, stdcolor="#FF9848")
    # regime_name = "STRONGPOLARIZEDLINE"
    # experiment_names = [f"{regime_name}_startNONpolarized_6Bots",
    #                     f"{regime_name}_startMIDpolarized_6Bots",
    #                     f"{regime_name}_startpolarized_6Bots"]
    # paths = [f'C:\\Users\\David\\Documents\\VisualSwarm\\controllers\\blank_controller\\simulation_data\\{i}' for i in experiment_names]
    # colors = ["#ffcccc", "#ffffb2", "#cce5cc"]
    # rad_limits = [1, 2, 3]
    # titles = [f"Initial max $\\Delta\\Phi={i}$ [rad]" for i in rad_limits]
    # plotting_tools.plot_COMvelocity_summary_perinit(paths, titles, colors, "Mean Center-of-mass velocity for different intial conditions in polarized line regime")

    #
    # center_of_mass = np.mean(data[:, :, [1, 2, 3], :], axis=1)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.scatter(data[0, :, 1, 100], data[0, :, 3, 100])
    # plt.scatter(center_of_mass[0, 0, 100], center_of_mass[0, 2, 100], s=10)
    # plt.show()

    # plotting data
    # plotting_tools.plot_velocities(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_distances(summary, data, DISTANCE_REFERENCE, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_orientation(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_iid(summary, data, 0)
    # #
    #
    min_iid = data_tools.calculate_min_iid(summary, data)
    pm = data_tools.calculate_ploarization_matrix(summary, data)
    plt.ion()
    for t in range(0, 35000, 5):
        dist = 1-pm[0, :, :, t].astype('float')
        print(dist.dtype)
        # sermat = compute_serial_matrix(1-pm[0, :, :, t].astype('float'))
        linkage_matrix = linkage(dist, "single")
        dendrogram(linkage_matrix, color_threshold=1, labels=[i for i in range(10)], show_leaf_counts=True)
        # plt.imshow(sermat[0])
        plt.draw()
        input()
        plt.clf()
    #
    # plotting_tools.plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848')
    # plotting_tools.plot_mean_iid_over_runs(summary, data, stdcolor='#FF9848')
    # plotting_tools.plot_min_iid_over_runs(summary, data, stdcolor="#FF9848")
    # plotting_tools.plot_mean_pol_over_runs(summary, data, stdcolor='#FF9848')
    #
    #
    # plotting_tools.plot_mean_ploarization(summary, data)
    # plotting_tools.plot_mean_iid(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)
    # plotting_tools.plot_min_iid(summary, data, changed_along=change_along, changed_along_alias=change_along_alias)

    #### LAST STEPS
    iid = data_tools.calculate_interindividual_distance(summary, data)
    mean_iid = np.mean(np.mean(iid, axis=1), axis=1)
    # plt.plot(mean_iid[0])

    iid_ratios = []
    for i in range(100):
        pol_ratio = np.count_nonzero(mean_iid<i*35) / mean_iid.shape[1]
        iid_ratios.append(pol_ratio)

    polrats_over_exps.append(np.array(iid_ratios))
    break

# plt.show()


fig, ax = plt.subplots(1, 2, figsize=[10, 5], sharey=True)
x = np.array([i*35 for i in range(100)])

large = np.vstack((polrats_over_exps[0], polrats_over_exps[1]))
mlarge = np.mean(large, axis=0)
stdlarge = np.std(large, axis=0)

# largebl = np.vstack((polrats_over_exps[6], polrats_over_exps[7]))
# mlargebl = np.mean(largebl, axis=0)
# stdlargebl = np.std(largebl, axis=0)
#
# small = np.vstack((polrats_over_exps[2], polrats_over_exps[3]))
# msmall = np.mean(small, axis=0)
# stdsmall = np.std(small, axis=0)
#
# smallbl = np.vstack((polrats_over_exps[4], polrats_over_exps[5]))
# msmallbl = np.mean(small, axis=0)
# stdsmallbl = np.std(small, axis=0)

plt.axes(ax[0])
plt.title('Low $\\beta_0$')
#error_band = plt.fill_between(x, mlarge-stdlarge, mlarge+stdlarge, alpha=0.5, edgecolor="#ffbcd9", facecolor="#ffbcd9")
plt.plot(x, mlarge, label="$R^{iid}_{thr-}$ with $L_{eq}^L$", color="#89bcff", linewidth="3")

#error_band = plt.fill_between(x, msmall-stdsmall, msmall+stdsmall, alpha=0.5, edgecolor="#ffcc89", facecolor="#ffcc89")
# plt.plot(x, msmall, label="$R^{iid}_{thr-}$ with $L_{eq}^S$", color="#ffcc89", linewidth="3")

# showing equilibrium distances
# ax[0].axvline(1000, 0, 1, label="$L_{eq}^L$", color="#89bcff", linestyle="-.")
# ax[0].axvline(741, 0, 1, label="$L_{eq}^S$", color="#ffcc89", linestyle="-.")
# ax[0].axvline(150, 0, 1, label="BL", color="gray", linewidth="1")



plt.xlabel('Distance Threshold [mm]')
plt.ylabel('Relative Time Below Threshold $R^{iid}_{thr-}$')

plt.axes(ax[1])
plt.title('High $\\beta_0$')
#error_band = plt.fill_between(x, mlargebl-stdlargebl, mlargebl+stdlargebl, alpha=0.5, edgecolor="#89bcff", facecolor="#89bcff")
# plt.plot(x, mlargebl, label="$R^{iid}_{thr-}$ with $L_{eq}^L$", color="#89bcff", linewidth="3")

#error_band = plt.fill_between(x, msmallbl-stdsmallbl, msmallbl+stdsmallbl, alpha=0.5, edgecolor="#ffcc89", facecolor="#ffcc89")
# plt.plot(x, msmallbl, label="$R^{iid}_{thr-}$ with $L_{eq}^S$", color="#ffcc89", linewidth="3")

# showing equilibrium distances
# ax[1].axvline(1000, 0, 1, label="$L_{eq}^L$", color="#89bcff", linestyle="-.")
# ax[1].axvline(741, 0, 1, label="$L_{eq}^S$", color="#ffcc89", linestyle="-.")
# ax[1].axvline(150, 0, 1, label="BL", color="gray", linewidth="1")

plt.xlabel('Distance Threshold [mm]')
plt.legend()

plt.show()
