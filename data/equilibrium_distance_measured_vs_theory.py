"""
@author: mezdahun
@description: extension script to hold experimental measurement of equilibrium distance results
              and a script to plot them.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

exp_template = {
    'gamma': 0.2,
    'v_0': 0,
    'alpha_0': 0.5,
    'alpha_1': 0.005,
    'alpha_2': 0,
    'beta_0': 0.5,
    'beta_1': 0.01,
    'beta_2': 0
}

# Experimental scenario 1 (19-02-2021)
exp_1 = exp_template.copy()
exp_1['alpha_1'] = np.array([0.0015, 0.00125, 0.001, 0.00095, 0.00075, 0.00065, 0.00045, 0.00035, 0.00025, 0.0001])
exp_1['theory_bl015'] = 0.15/(2*exp_1['alpha_1'])
exp_1['measured_sim'] = np.array([52, 61, 77, 79, 102, 119, 169, 219, 331, 605])
exp_1['measured_sim_resc129'] = np.array([67, 78, 99, 102, 132, 156, 220, 292, 431, 781])
exp_1['measured_exp'] = np.array([70, 84, 100, 113, 139, 151, 222, 267, 389, 590])
exp_1['fitted_sim_bl0155'] = 0.155/(2*exp_1['alpha_1'])
exp_1['fitted_exp_bl020'] = 0.2/(2*exp_1['alpha_1'])
exp_1['eq_distance'] = np.array([250, 190, 135, 104, 74, 68, 51, 48, 45, 38, 34, 30])


fig, ax = plt.subplots(1, 2, figsize=[10, 8], sharey=True)
plt.axes(ax[0])
plt.plot(exp_1['alpha_1'], exp_1['measured_exp'], color='#666666', linewidth=6, label='Measurement (exp)')
plt.plot(exp_1['alpha_1'], exp_1['fitted_exp_bl020'], color='#ffc4c4', linewidth=4, ls='--', label='Theory BL=20cm')
plt.plot(exp_1['alpha_1'], exp_1['measured_sim'], color='#000000', linewidth=6, label='Measurement (sim)')
plt.plot(exp_1['alpha_1'], exp_1['fitted_sim_bl0155'], color='#ff7676', linewidth=4, ls='--', label='Theory BL=15.5cm')


ax[0].set_yscale('log')
ax[0].set_yticks([50, 100, 250, 500, 1000])
ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.title(r'Dependence of equilibrium distance on model parameter $\alpha_1$')
plt.xlabel(r'$\alpha_1$ [AU]')
plt.ylabel('Equilibrium distance [cm]')
plt.legend()
plt.grid(which='major', axis='y')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.axes(ax[1])
plt.plot(exp_1['alpha_1'], exp_1['measured_exp'], color='#666666', linewidth=6, label='Measurement (exp)')
plt.plot(exp_1['alpha_1'], exp_1['fitted_exp_bl020'], color='#ffc4c4', linewidth=4, ls='--', label='Theory BL=20cm')
plt.plot(exp_1['alpha_1'], exp_1['measured_sim_resc129'], color='#000000', linewidth=2, label='Measurement (sim resc.)')

# ax[1].set_yscale('log')
ax[1].set_yticks([50, 100, 250, 500, 1000])
# ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.title(r'Matched Equilibrium distances')
plt.xlabel(r'$\alpha_1$ [AU]')
# plt.ylabel('Equilibrium distance [cm]')
plt.legend()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# plt.plot(exp_1['alpha_1'], exp_1['theory_bl015'], color='green', marker='.', linewidth=0, markersize=16)
# plt.plot(exp_1['alpha_1'], exp_1['theory_bl015'], color='green', marker='.', linewidth=0, markersize=16)
# plt.plot(exp_1['alpha_1'], exp_1['theory_bl015'], color='green', marker='.', linewidth=0, markersize=16)
# plt.plot(exp_1['alpha_1'], exp_1['theory_bl015'], color='green', marker='.', linewidth=0, markersize=16)
plt.grid(which='major', axis='y')
plt.show()
