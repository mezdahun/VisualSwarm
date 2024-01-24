"""This script contains the main softeare tools to read back
data exported by the exploration tool (written in Julia) to explore tracking
data of the visual swarm experiments."""

import pandas as pd
import os
import numpy as np

INPUT_FOLDER = './input_data'

# Reading back robot data from robots.parquet
robots = pd.read_parquet(os.path.join(INPUT_FOLDER, 'robots.parquet'), engine='pyarrow')
# defining time axis
t = robots['t']
# undersampling time by 10
t_u = t[::10]
# Reading metrics data
metrics = pd.read_parquet(os.path.join(INPUT_FOLDER, 'metrics.parquet'), engine='pyarrow')

# # exploring data that have been reead back
# print(metrics.head())
# print(metrics.columns)
# print(metrics.shape)
# print(metrics.dtypes)
#
# # plotting all metrics in a sublots with title and y axis label
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
# # choosing some pastel colors as many as the number of metrics
# # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
# colors = ['xkcd:light blue', 'xkcd:light orange', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple',
#           'xkcd:light yellow', 'xkcd:light pink', 'xkcd:light brown']
# for i, col in enumerate(metrics.columns):
#     ax[i // 2, i % 2].plot(t_u, metrics[col], color=colors[i])
#     ax[i // 2, i % 2].set_title(col)
#     ax[i // 2, i % 2].set_ylabel(col)
#     plt.xlabel('time (s)')
# plt.show()
#
#
#
# exploring data that have been reead back
print(robots.head())
print(robots.columns)
print(robots.shape)
print(robots.dtypes)
num_robots = np.max(robots['robot_id'])
#
# # plotting all metrics in a sublots with title and y axis label
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
# # choosing 20 pastel colors as many as the number of metrics
# # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
# colors = ['xkcd:light blue', 'xkcd:light orange', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple',
#           'xkcd:light yellow', 'xkcd:light pink', 'xkcd:light brown', 'xkcd:light blue', 'xkcd:light orange',
#           'xkcd:light green', 'xkcd:light red', 'xkcd:light purple', 'xkcd:light yellow', 'xkcd:light pink',
#           'xkcd:light brown', 'xkcd:light blue', 'xkcd:light orange', 'xkcd:light green']
# for i, col in enumerate(robots.columns):
#     ax[i // 4, i % 4].plot(t, robots[col], color=colors[i])
#     ax[i // 4, i % 4].set_title(col)
#     ax[i // 4, i % 4].set_ylabel(col)
#     plt.xlabel('time (s)')
# plt.show()

# Reading derived metrics coming from julia with HDF5
import h5py

f = h5py.File(os.path.join(INPUT_FOLDER, 'derived.jld2'), 'r')
print(list(f.keys()))
print(f['center_of_mass'].shape)
print(f['center_of_mass'].dtype)
print(f['distance_matrices'].shape)
print(f['distance_matrices'].dtype)
print(f['furthest_robots'].shape)
print(f['furthest_robots'].dtype)

# defining center of mass
COM_x = f['center_of_mass'][:, 0]
COM_y = f['center_of_mass'][:, 1]

# calculate mean distance:
import numpy as np

mean_dist = np.mean(np.mean(f['distance_matrices'], axis=-1), axis=-1)
# calculate distance std
std_dist = np.mean(np.std(f['distance_matrices'], axis=-1), axis=-1)

# # plotting the mean distance with error shade
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(t_u, mean_dist, label='mean')
# plt.fill_between(t_u, mean_dist - std_dist, mean_dist + std_dist, alpha=0.5, label='std')
# # labeling and annotation
# plt.xlabel('time (s)')
# plt.ylabel('mean distance (mm)')
# plt.title('Mean distance between robots')
# plt.legend()
# plt.show()

# reading back convcex hulls from json file
import json
with open(os.path.join(INPUT_FOLDER, 'convex_hull.json'), 'r') as f:
    convex_hulls = json.load(f)

# Creating a video with matplotlib where in each timestep in an empty white frame we draw the convex hull in that
# timestep and put a green dot in the center of mass of the swarm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np

# defining the figure
fig, ax = plt.subplots(figsize=(10, 10))
# defining the plot limits
ax.set_xlim(-6000, 6000)
ax.set_ylim(-6000, 6000)
# defining the plot title
ax.set_title('Convex hull of the swarm')
# defining the plot labels
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
# defining the plot background
ax.set_facecolor('white')
# defining the plot grid
ax.grid(True)
# defining the plot aspect ratio
ax.set_aspect('equal')

# defining the plot elements
# defining the convex hull plot
convex_hull_plot, = ax.plot([], [], color='black', label='convex hull')
# defining the center of mass plot
center_of_mass_plot, = ax.plot([], [], 'go', label='center of mass')
# defining scatter plot for 10 red crosses for robots
robot_plot, = ax.plot([], [], 'rx', label='robots')


# for i, hull in enumerate(convex_hulls):
#     hull_x = [p[0] for p in hull]
#     hull_y = [p[1] for p in hull]
#     convex_hull_plot.set_data(hull_x, hull_y)
#     center_of_mass_plot.set_data(COM_x[i], COM_y[i])
#     time_text.set_text(f'time: {t_u[i]:.2f} s')
#     frame_number_text.set_text(f'frame: {i}')
#     plt.savefig(f'./output_data/convex_hull_{i:04d}.png')

# reshape robot data into matrix of shape (num_robots, num_timesteps)
raw_robot_x = np.array(robots['x'])
robot_ids = np.array(robots['robot_id'])
# creating a matrix of shape (num_robots, num_timesteps) where each row is the x coordinate of a robot
robot_x = np.zeros((num_robots, len(t_u)))
# populating the matrix according to robot_ids
for i in range(num_robots):
    robot_x[i, :] = raw_robot_x[robot_ids == i+1]

raw_robot_y = np.array(robots['z'])
robot_y = np.zeros((num_robots, len(t_u)))
for i in range(num_robots):
    robot_y[i, :] = raw_robot_y[robot_ids == i+1]

# being sure that the data is correctly populated
for i in range(num_robots):
    assert np.all(robot_x[0, :] == raw_robot_x[robot_ids == 1])

# defining the animation function
def animate(i):
    hull = convex_hulls[i]
    hull_x = [p[0] for p in hull]
    hull_x.append(hull_x[0])
    hull_y = [p[1] for p in hull]
    hull_y.append(hull_y[0])
    convex_hull_plot.set_data(hull_x, hull_y)
    center_of_mass_plot.set_data(COM_x[i], COM_y[i])
    # showing robots with red crosses
    xs = []
    ys = []
    for ri in range(num_robots):
        # adding x and y coordinate to plotting data
        xs.append(robot_x[ri, i])
        ys.append(robot_y[ri, i])
    robot_plot.set_data(xs, ys)
    print(f'frame: {i}')
    return convex_hull_plot, center_of_mass_plot

# defining the animation taking every 100th frame until 4000
anim = FuncAnimation(fig, animate, frames=np.arange(0, 110000, 500))

# saving to m4 using ffmpeg writer
#writervideo = animation.FFMpegWriter(fps=10)
# anim.save(os.path.join(INPUT_FOLDER, "convex_hulls.mp4"), writer=writervideo, )
f = os.path.join(INPUT_FOLDER, "convex_hulls.gif")
writergif = animation.PillowWriter(fps=10)
anim.save(f, writer=writergif)
plt.close()



