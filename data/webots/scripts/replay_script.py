"""
@description: cPOC for processing optitrack data for publication.
"""
import time

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
import math
from scipy.cluster.hierarchy import linkage
from visualswarm.simulation_tools import data_tools, plotting_tools

# if data is freshly created first summarize it into multidimensional array
EXPERIMENT_NAMES = ["TestAfterLongPause_An1.5_Bn5_10bots_FOV3.455751918948773"]
DISTANCE_REFERENCE = [0, 0, 0]

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

polrats_over_exps = []
for expi in range(len(EXPERIMENT_NAMES)):
    data_path = f'/mnt/DATA/mezey/Seafile/SwarmRobotics/VisualSwarm Simulation Data/RealExperiments_Exploration_10bots/{EXPERIMENT_NAMES[expi]}'
    # if data is freshly created first summarize it into multidimensional array
    change_along = None
    change_along_alias = None

    # retreiving data
    data_tools.summarize_experiment(data_path, EXPERIMENT_NAMES[expi], skip_already_summed=True)
    summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAMES[expi])

    runi = 1
    iidm = data_tools.calculate_interindividual_distance(summary, data)[runi, ...]
    pm = data_tools.calculate_ploarization_matrix(summary, data)
    plt.ion()
    fig, ax = plt.subplots(2, 1)
    data[:, :, [1, 2, 3], :] = data[:, :, [1, 2, 3], :]*1000
    for t in range(0, 35000, 30):
        plt.clf()
        plt.axes(ax[0])
        niidm = (iidm[:, :, t] - np.min(iidm[:, :, t])) / (np.max(iidm[:, :, t]) - np.min(iidm[:, :, t]))
        dist = (1 - pm[runi, :, :, t].astype('float') + niidm) / 2
        # sermat = compute_serial_matrix(1-pm[0, :, :, t].astype('float'))
        linkage_matrix = linkage(dist, "single")
        ret = dendrogram(linkage_matrix, color_threshold=1.2, labels=[i for i in range(10)], show_leaf_counts=True)

        plt.axes(ax[1])
        center_of_mass = np.mean(data[:, :, [1, 2, 3], :], axis=1)
        colors = [color for _, color in sorted(zip(ret['leaves'], ret['leaves_color_list']))]
        plt.scatter(data[runi, :, 1, t], data[runi, :, 3, t], s=100, c=colors)
        for i in range(10):
            plt.annotate(i, (data[runi, i, 1, t], data[runi, i, 3, t] + 0.2))
        # print("x: ", data[runi, :, 1, t])
        # print("y: ", data[runi, :, 2, t])
        # print("z: ", data[runi, :, 3, t])
        # print("ori: ", data[runi, :, 4, t])
        ori = data[runi, :, 4, t]
        ms = 200
        for ri in range(len(ori)):
            x = data[runi, :, 1, t]
            y = data[runi, :, 3, t]
            angle = ori[ri]
            # print(angle)
            plt.arrow(x[ri], y[ri], ms * math.cos(angle), ms * math.sin(angle), color="white")
        plt.scatter(center_of_mass[runi, 0, t], center_of_mass[runi, 2, t], s=50)
        # Show arena borders
        circle = plt.Circle((0, 0), 3000, color='black', fill=False)
        ax[1].add_patch(circle)

        plt.xlim(-3000, 3000)
        plt.ylim(-2000, 2000)

        # plt.draw()
        # re-drawing the figure
        fig.canvas.draw()
          # to flush the GUI events
        fig.canvas.flush_events()
        time.sleep(0.05)

plt.show()
