import os

from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from visualswarm.simulation_tools import data_tools, plotting_tools

BATCH_NAME = "TESTAFTERMERGE_Exploration_10bots"
EXPERIMENT_FOLDER = f"/mnt/DATA/mezey/Seafile/SwarmRobotics/VisualSwarm Simulation Data/{BATCH_NAME}"
iid_path = os.path.join(EXPERIMENT_FOLDER, "iid.npy")
pol_path = os.path.join(EXPERIMENT_FOLDER, "pol.npy")

alphas = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 3, 5]
bethas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 3, 5]
num_runs = 3

if not os.path.isfile(iid_path) or not os.path.isfile(pol_path):
    iid_matrix = np.zeros((num_runs, len(alphas), len(bethas)))
    pol_matrix = np.zeros((num_runs, len(alphas), len(bethas)))

    donei = 0
    num_data_points = len(alphas)*len(bethas)
    for ai, alpha in enumerate(alphas):
        for bi, beta in enumerate(bethas):
            print(f"Experiment with alpha: {alpha} and beta {beta}")
            EXPERIMENT_NAME = f"TestAfterLongPause_An{alpha}_Bn{beta}_10bots_FOV3.455751918948773"
            data_path = f'{EXPERIMENT_FOLDER}/{EXPERIMENT_NAME}'
            data_tools.summarize_experiment(data_path, EXPERIMENT_NAME, skip_already_summed=True)
            summary, data = data_tools.read_summary_data(data_path, EXPERIMENT_NAME)
            mean_iid = data_tools.calculate_mean_iid(summary, data)
            mean_pol = data_tools.calculate_mean_polarization(summary, data)
            iid_matrix[:, ai, bi] = mean_iid
            pol_matrix[:, ai, bi] = mean_pol
            donei += 1
            print(f"Process: {donei/num_data_points*100}%")

    print("Saving final IID and POL matrices")
    np.save(iid_path, iid_matrix)
    np.save(pol_path, pol_matrix)
else:
    print("Previously saved data found, loading IID and POL matrices!")
    iid_matrix = np.load(iid_path)
    pol_matrix = np.load(pol_path)

miid_matrix = np.mean(iid_matrix, axis=0)
mpol_matrix = np.mean(pol_matrix, axis=0)


plt.imshow(miid_matrix)
plt.figure()
plt.imshow(mpol_matrix)
plt.show()