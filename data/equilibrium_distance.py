"""
@author: mezdahun
@description: extension script to hold experimental measurement of equilibrium distance results
              and a script to plot them.
"""

exp_template = {
    'gamma': 0.2,
    'v_0': 0,
    'alpha_0': 0.5,
    'alpha_1': [0.005],
    'alpha_2': 0,
    'beta_0': 0.5,
    'beta_1': 0.01,
    'beta_2': 0,
    'eq_distance': [None]
}

# Experimental scenario 1 (19-02-2021)
exp_1 = exp_template.copy()
exp_1['alpha_1'] = [0.001, 0.003, 0.004, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080, 0.100]
exp_1['eq_distance'] = [250, 190, 135, 104, 74, 68, 51, 48, 45, 38, 34, 30]


def plot_eq_distance(exp):
    """Plotting equilibrium distance according to an experimental data dict defined above"""
    pass
