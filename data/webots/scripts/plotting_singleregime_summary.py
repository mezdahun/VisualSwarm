from visualswarm.simulation_tools import data_tools, plotting_tools

regime_name = "POLARIZEDLINE"

experiment_names = [f"{regime_name}_startNONpolarized_6Bots",
                    f"{regime_name}_startMIDpolarized_6Bots",
                    f"{regime_name}_startpolarized_6Bots"]
paths = [f'C:\\Users\\David\\Documents\\VisualSwarm\\controllers\\blank_controller\\simulation_data\\{i}' for i in experiment_names]

# for i, path in enumerate(paths):
#     print(f"Summarizing experiment {experiment_names[i]}")
#     data_tools.summarize_experiment(path, experiment_names[i])

colors = ["#ffcccc", "#ffffb2", "#cce5cc"]
rad_limits = [1, 2, 3]
titles = [f"Initial max $\\Delta\\Phi={i}$ [rad]" for i in rad_limits]

print("Preparing plot...")
plotting_tools.plot_COMvelocity_summary_perInit(paths, titles, colors,
                                                "Mean Center-of-mass velocity for different intial conditions in "
                                                "polarized line regime")
print("Done")