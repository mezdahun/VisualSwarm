"""In some cases the agents rotation is permanently shifted from their actual heading angle. Whit this script we read
the summaryd.json file and add the shift to the heading angle of the selected agents, then save the postprocessed file"""

import numpy as np


# opening filedialog window to choose file ending with _summaryd.npy
def open_file(file_ending='_summaryd.npy'):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if file_path.endswith(file_ending):
        return file_path
    else:
        print('Please choose a file ending with', file_ending)
        return open_file(file_ending)


# loading the summaryd file
def load_summaryd(file_path):
    summaryd = np.load(file_path, allow_pickle=True)
    return summaryd

# ## Test which agent is which:
# selected_agents = [i for i in range(10)]
# selected_rotation_shift = np.array([0.5 for i in range(10)]) * np.pi  # in radians, positive left, negative right
#
# file_path = open_file()
# summaryd = load_summaryd(file_path)
#
# for i, agent in enumerate(selected_agents):
#     summaryd_ = summaryd.copy()
#     summaryd_[0, agent, 4, :] += selected_rotation_shift[i]
#
#     # adding the term "postprocess" to the file name
#     file_path_new = file_path.split('summaryd.npy')[0] + f'postprocess_a{agent}_summaryd.npy'
#     np.save(file_path_new, summaryd_)

selected_agents = [4]
selected_rotation_shift = np.array([0.1]) * np.pi  # in radians

file_path = open_file()
summaryd = load_summaryd(file_path)
for i, agent in enumerate(selected_agents):
    summaryd[0, agent, 4, :] += selected_rotation_shift[i]

# adding the term "postprocess" to the file name
file_path = file_path[:-4] + '_postprocess.npy'
np.save(file_path, summaryd)


