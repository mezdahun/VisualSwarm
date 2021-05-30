from visualswarm.simulation_tools import data_tools, plotting_tools
import os

regime_names = ["STRONGPOLARIZEDLINE", "POLARIZEDDISC", "SWARMFLUID", "POLARIZEDSWARM"]
basepath = "C:\\Users\\David\\Documents\\VisualSwarm\\controllers\\blank_controller\\simulation_data"

regime_dict = {}

for regime_name in regime_names:

    experiment_names = [f"{regime_name}_startNONpolarized_6Bots",
                        f"{regime_name}_startMIDpolarized_6Bots",  # [f"{regime_name}_startNONpolarized_6Bots",
                        f"{regime_name}_startpolarized_6Bots"]
    paths = [os.path.join(basepath, i) for i in experiment_names]
    regime_dict[regime_name] = {"paths": paths}

# If summaries are not yet ready use:
for regime_name, path_dict in regime_dict.items():
    print(f"Summarizing experiments for regime {regime_name}")
    for path in path_dict["paths"]:
        print(f"Summarizing experiment {os.path.basename(path)}")
        data_tools.summarize_experiment(path, os.path.basename(path))

colors = ["#ffcccc", "#ffffb2", "#cce5cc"]
rad_limits = [1, 2, 3]
titles = ["Polarized Line", "Polarized Disk", "Swarm Fluid", "Polarized Swarm"]

print("Preparing COM velocity plot...")
plotting_tools.plot_COMvelocity_summary_perRegimeandInit(regime_dict, titles, colors,
                                                         "Mean Center-of-mass velocity for different collective "
                                                         "motion regimes")
print("Preparing min I-I.D. plot...")
plotting_tools.plot_min_iid_summary_perRegimeandInit(regime_dict, titles, colors,
                                                     "Mean of minimum inter-individual distances over "
                                                     "motion regimes")

print("Preparing Heading angle polarization plot...")
plotting_tools.plot_mean_pol_summary_perRegimeandInit(regime_dict, titles, colors,
                                                     "Mean of heading angle polarization "
                                                     "over different motion regimes")
print("Done")
