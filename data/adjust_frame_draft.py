import numpy as np
import cv2
import matplotlib.pyplot as plt


## Adjusting image to center and resize (after detection)
left_adjust = 0
right_adjust = 28
#
for imi in range(1, 2):
    img = cv2.imread(f'{imi}.jpg')
    orig_shape = (img.shape[1], img.shape[0])
    cropped = img[:, left_adjust:-right_adjust]
    final = cv2.resize(cropped, orig_shape)



    ## Rescale image with reverse horizontal mapping function
    orig_img_width = orig_shape[0]

    FOV = 3.8
    phi_start = - (FOV / 2)
    phi_end = FOV / 2
    a = 0.8*np.pi
    h_domain = np.linspace(phi_start, phi_end, orig_img_width)
    non_lin_part = a * np.square(h_domain)
    lin_part = np.ones(orig_img_width)*3
    h_reverse_mapping = np.maximum(non_lin_part-1.5, lin_part)
    discrete_h_reverse_mapping = np.array([np.round(num) for num in h_reverse_mapping])
    if imi == 1:
        plt.plot(h_domain, discrete_h_reverse_mapping)
        plt.show()

    new_width = np.sum(discrete_h_reverse_mapping)
    new_shape = (int(new_width), orig_shape[1])
    new_img = cv2.resize(final, new_shape)

    done_respe_i = 0
    for i in range(orig_img_width):
        index_end = done_respe_i + discrete_h_reverse_mapping[i]
        for j in range(int(discrete_h_reverse_mapping[i])):
            new_img[:, int(done_respe_i+j)] = final[:, i]
        done_respe_i = index_end

    cv2.imshow('image', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


fig, ax = plt.subplots(2, 1)
for k in range(32):
    detection = np.zeros(int(orig_img_width))
    detection[k*10:k*10+8] = 1

    rescaled_detection = np.zeros(int(new_width))
    done_respe_i = 0
    for i in range(orig_img_width):
        index_end = done_respe_i + discrete_h_reverse_mapping[i]
        for j in range(int(discrete_h_reverse_mapping[i])):
            rescaled_detection[int(done_respe_i+j)] = detection[i]
        done_respe_i = index_end

    orig_domain = h_domain
    new_domain = np.linspace(phi_start, phi_end, int(new_width))
    plt.axes(ax[0])
    plt.plot(orig_domain, detection)
    plt.axes(ax[1])
    plt.plot(new_domain, rescaled_detection)
plt.show()


