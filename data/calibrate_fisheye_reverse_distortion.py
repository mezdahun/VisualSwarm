"""
Short script to calibrate reverse distortion parameters of fisheye lenses.
To use, simply put example images in the data folder and run the script to see what the visual distortion
with parameters given in contrib.vision does to the horizontal axis. Then tune a_lin, a_nonlin and offset_lin, and
the centering offets on the left and right

The outut is images comparing original with processed in the same folder.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from visualswarm.vision import vprocess
from datetime import datetime
from glob import glob


# ## Adjusting image to center and resize (after detection)
images = glob('./*.jpg')
imi = 0
for imp in images:
    img = cv2.imread(imp)
    centered = vprocess.center_fisheye_circle(img, 'Robot1')
    remapped = vprocess.correct_fisheye_approx(centered, 'Robot1')

    cv2.imwrite(f'{imi}.jpg', np.hstack((img, remapped)))
    imi += 1


orig_img_width=320
fig, ax = plt.subplots(2, 1)
for k in range(32):
    detection = np.zeros(int(orig_img_width))
    detection[k*10:k*10+8] = 1

    t0 = datetime.now()
    centered_detection = vprocess.center_fisheye_circle(detection, 'Robot1')
    rescaled_detection = vprocess.correct_fisheye_approx(centered_detection, 'Robot1')#np.zeros(int(new_width))
    t1 = datetime.now()
    print((t1-t0).total_seconds())

    FOV = 3.8
    phi_start = - (FOV / 2)
    phi_end = FOV / 2
    orig_domain = np.linspace(phi_start, phi_end, orig_img_width)
    # new_domain = np.linspace(phi_start, phi_end, int(new_width))
    plt.axes(ax[0])
    plt.plot(orig_domain, detection)
    plt.axes(ax[1])
    plt.plot(orig_domain, [round(det) for det in rescaled_detection])

plt.show()


