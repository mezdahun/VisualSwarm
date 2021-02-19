"""
@author: mezdahun
@description: extension script to hold experimental measurement of equilibrium distance results
              and a script to plot them.
"""
import matplotlib.pyplot as plt
import numpy as np

exp_template = {
    'gamma': 0.2,
    'v_0': 0,
    'alpha_0': 0.5,
    'alpha_1': 0.005,
    'alpha_2': 0,
    'beta_0': 0.5,
    'beta_1': 0.01,
    'beta_2': 0,
    'eq_distance': None
}

# Experimental scenario 1 (19-02-2021)
exp_1 = exp_template.copy()
exp_1['alpha_1'] = np.array([0.001, 0.003, 0.004, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080, 0.100])
exp_1['eq_distance'] = np.array([250, 190, 135, 104, 74, 68, 51, 48, 45, 38, 34, 30])


def plot_measurement(exp, x_key, y_key, x_label, y_label, title):
    """Plotting equilibrium distance according to an experimental data dict defined above"""
    xax = exp[x_key]
    yax = exp[y_key]
    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    plt.xlim([xax[0], xax[-1]])
    plt.plot(xax, yax, color='green', marker='.', linewidth=0, markersize=16)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


plot_measurement(exp_1, 'alpha_1', 'eq_distance',
                 r'$\alpha_1$ [AU]', 'Equilibrium distance [cm]',
                 r'Dependence of equilibrium distance on model parameter $\alpha_1$')
