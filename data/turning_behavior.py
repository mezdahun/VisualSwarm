import numpy as np

# BET0=1, BET1=0.001

angles = np.array([-67.5, -45, -22.5, 0, 22.5, 45, 67.5])
distances = np.array([50, 75, 100, 125, 150, 200, 250])
exp_dpsi = np.array([
    [-0.085, -0.001, 0.034, 0.058, 0.082, 0.106, 0.094],
    [-0.2, -0.045, 0.009, 0.03, 0.051, 0.05, 0.097],
    [-0.184, -0.064, -0.026, -0.014, 0.011, 0.036, 0.059],
    [-0.005, -0.001, 0.002, 0.001, -0.001, -0.004, -0.005],
    [0.187, 0.075, 0.024, 0, 0, -0.024, -0.051],
    [0.249, 0.09, 0.012, -0.01, -0.033, -0.079, -0.102],
    [0.157, 0.013, 0.011, -0.035, -0.082, -0.106, -0.13]])

sim_dpsi = np.array([
    [-0.281, -0.1189, -0.0724, -0.0142, 0.009, 0.0556, 0.0789],
    [-0.2131, -0.1132, -0.0484, -0.0209, 0.00669, 0.0344, 0.0529],
    [-0.1179, -0.0575, -0.027, -0.0119, -0.0019, 0.0186, 0.028],
    [0, -0.002, 0, 0, -0.0011, -0.0018, -0.002],
    [0.1227, 0.0572, 0.0317, 0.0106, -0.0051, -0.0264, -0.037],
    [0.2243, 0.1131, 0.0578, 0.02011, -0.0081, -0.0364, -0.0552],
    [0.2934, 0.1425, 0.0607, 0.025, -0.0096, -0.0448, -0.0683]])

exp_dpsi_corr = np.array([
    [-0.346, -0.132, -0.052, -0.037, 0.057, 0.045, 0.022],
    [-0.24, -0.072, -0.033, 0.008, 0.038, 0.058, 0.0707],
    [-0.169, -0.079, -0.027, -0.008, 0.016, 0.037, 0.0508],
    [0.014, 0.009, 0.0016, 0.001, -0.0008, -0.001, -0.008],
    [0.156, -0.0009, 0.0208, 0.0068, -0.013, -0.098, -0.042],
    [0.237, 0.073, 0.0326, -0.031, -0.062, -0.073, -0.072],
    [0.277, 0.109, 0.0494, 0.001, -0.01, -0.058, -0.058]])

axes = []

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(int(f'131'), projection='3d')
axes.append(ax)
#
x, y = np.meshgrid(angles, distances)
#
# p = ax.plot_surface(x, y, undes_matrix.T, cmap="seismic", alpha=0.5)#, #vmin=0, vmax=3750)
p = ax.plot_surface(x, y, exp_dpsi.T, cmap="seismic", alpha=0.5, vmin=-0.3, vmax=0.3)
# ax.set_zticks([-0.3, 0, 0.3])
# ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])
ax.set_xticks(angles)
ax.set_yticks([50, 100, 150, 200, 250])
ax.set_xlabel('angle [deg]')
ax.set_ylabel('distance [cm]')
plt.axes(ax)
# plt.title(f'Turning response (exp)')

ax = fig.add_subplot(int(f'132'), projection='3d')
axes.append(ax)
#
x, y = np.meshgrid(angles, distances)
#
# p = ax.plot_surface(x, y, undes_matrix.T, cmap="seismic", alpha=0.5)#, #vmin=0, vmax=3750)
p = ax.plot_surface(x, y, sim_dpsi.T, cmap="seismic", alpha=0.5, vmin=-0.3, vmax=0.3)
# ax.set_zticks([-0.3, 0, 0.3])
# ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])
ax.set_xticks(angles)
ax.set_yticks([50, 100, 150, 200, 250])
ax.set_xlabel('angle [deg]')
ax.set_ylabel('distance [cm]')
plt.axes(ax)
# plt.title(f'Turning response (sim)')

ax = fig.add_subplot(int(f'133'), projection='3d')
axes.append(ax)
#
x, y = np.meshgrid(angles, distances)
#
# p = ax.plot_surface(x, y, undes_matrix.T, cmap="seismic", alpha=0.5)#, #vmin=0, vmax=3750)
p = ax.plot_surface(x, y, exp_dpsi_corr.T, cmap="seismic", alpha=0.5, vmin=-0.3, vmax=0.3)
# ax.set_zticks([-0.3, 0, 0.3])
# ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])
ax.set_xticks(angles)
ax.set_yticks([50, 100, 150, 200, 250])
ax.set_xlabel('angle [deg]')
ax.set_ylabel('distance [cm]')
plt.axes(ax)


def on_move(event):
    for ax in axes:
        ax.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)
        #ax.set_zlim3d(0, 0.6)
    fig.canvas.draw_idle()

c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.suptitle('Comparison of socially induced turning response')

cax,kw = mpl.colorbar.make_axes([ax for ax in axes])
cbar = plt.colorbar(p, cax=cax, **kw)
cbar.set_label('$d\\psi$ [rad]', rotation=270)
cbar.set_ticks([-0.3, 0, 0.3])
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
plt.show()