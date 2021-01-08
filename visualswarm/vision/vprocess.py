"""
@author: mezdahun
@description: Processing low-level input to get High level input
"""
import datetime
import logging
from math import floor

import cv2
import numpy as np

from visualswarm.monitoring import ifdb
from visualswarm.contrib import camera, projection, segmentation, visual

# using main logger
logger = logging.getLogger('visualswarm.app')


def nothing(x):
    pass


def high_level_vision(raw_vision_stream, high_level_vision_stream, visualization_stream=None,
                      target_config_stream=None):
    """
    Process to process raw vision into high level vision and push it to a dedicated stream so that other behavioral
    processes can consume this stream
        Args:
            raw_vision_stream: multiprocessing.Queue type object to read raw visual input.
            high_level_vision_stream: multiprocessing.Queue type object to push high-level visual data.
            visualization_stream: stream to visualize raw vs processed vision, and to tune parameters interactively
        Returns:
            -shall not return-
    """
    hsv_low = segmentation.HSV_LOW
    hsv_high = segmentation.HSV_HIGH

    while True:
        (img, frame_id) = raw_vision_stream.get()
        # logger.info(raw_vision_stream.qsize())
        if visual.FIND_COLOR_INTERACTIVE:
            if target_config_stream is not None:
                if target_config_stream.qsize() > 1:
                    (R, B, G, hue_range, sv_min, sv_max) = target_config_stream.get()
                    target_hsv = cv2.cvtColor(np.uint8([[[B, G, R]]]), cv2.COLOR_BGR2HSV)
                    hsv_low = np.uint8([target_hsv[0][0][0] - hue_range, sv_min, sv_min])
                    hsv_high = np.uint8([target_hsv[0][0][0] + hue_range, sv_max, sv_max])

        # logger.info(raw_vision_stream.qsize())
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsvimg, hsv_low, hsv_high)

        # Gaussian blur
        blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        blurred = cv2.medianBlur(blurred, 9)

        conts, h = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        threshold_area = 50  # threshold area keep only larger contours
        fconts = [cnt for cnt in conts if cv2.contourArea(cnt) >= threshold_area]

        hull_list = []
        for i in range(len(fconts)):
            hull = cv2.convexHull(fconts[i])
            hull_list.append(hull)

        cv2.drawContours(img, fconts, -1, (0, 0, 255), 3)
        cv2.drawContours(img, hull_list, -1, (0, 255, 0), 3)
        cv2.drawContours(blurred, hull_list, -1, (255, 255, 255), -1)

        high_level_vision_stream.put((img, blurred, frame_id))
        if visualization_stream is not None:
            visualization_stream.put((img, blurred, frame_id))


def visualizer(visualization_stream, target_config_stream=None):
    """
    Process to Visualize Raw and Processed camera streams via a visualization stream. It is also used to tune parameters
    interactively, in this case a configuration stream is also used to fetch interactively given parameters from the user.
        Args:
            visualization_stream: stream to visualize raw vs processed vision, and to tune parameters interactively
            target_config_stream: stream to transmit segmentation parameters between interactive tuning input window and
                the visualization_stream
        Returns:
            -shall not return-
    """
    if visualization_stream is not None:
        if visual.FIND_COLOR_INTERACTIVE:
            cv2.namedWindow("Segmentation Parameters")
            cv2.createTrackbar("R", "Segmentation Parameters", segmentation.TARGET_RGB_COLOR[0], 255, nothing)
            cv2.createTrackbar("G", "Segmentation Parameters", segmentation.TARGET_RGB_COLOR[1], 255, nothing)
            cv2.createTrackbar("B", "Segmentation Parameters", segmentation.TARGET_RGB_COLOR[2], 255, nothing)
            cv2.createTrackbar("H_range", "Segmentation Parameters", segmentation.HSV_HUE_RANGE, 255, nothing)
            cv2.createTrackbar("SV_min", "Segmentation Parameters", segmentation.SV_MINIMUM, 255, nothing)
            cv2.createTrackbar("SV_max", "Segmentation Parameters", segmentation.SV_MAXIMUM, 255, nothing)
            color_sample = np.zeros((200, 200, 3), np.uint8)

        while True:
            # visualization
            (img, mask, frame_id) = visualization_stream.get()
            if visual.FIND_COLOR_INTERACTIVE:
                if target_config_stream is not None:
                    B = cv2.getTrackbarPos("B", "Segmentation Parameters")
                    G = cv2.getTrackbarPos("G", "Segmentation Parameters")
                    R = cv2.getTrackbarPos("R", "Segmentation Parameters")
                    color_sample[:] = [B, G, R]
                    HSV_HUE_RANGE = cv2.getTrackbarPos("H_range", "Segmentation Parameters")
                    SV_MINIMUM = cv2.getTrackbarPos("SV_min", "Segmentation Parameters")
                    SV_MAXIMUM = cv2.getTrackbarPos("SV_max", "Segmentation Parameters")
                    target_config_stream.put((R, B, G, HSV_HUE_RANGE, SV_MINIMUM, SV_MAXIMUM))
            vis_width = floor(camera.RESOLUTION[0] / visual.VIS_DOWNSAMPLE_FACTOR)
            vis_height = floor(camera.RESOLUTION[1] / visual.VIS_DOWNSAMPLE_FACTOR)
            cv2.imshow("Object Contours", cv2.resize(img, (vis_width, vis_height)))
            cv2.imshow("Final Area", cv2.resize(mask, (vis_width, vis_height)))
            if visual.FIND_COLOR_INTERACTIVE:
                cv2.imshow("Segmentation Parameters", color_sample)
            cv2.waitKey(1)
    else:
        logger.info('Visualization stream is None, visualization process returns!')


def FOV_extraction(high_level_vision_stream, FOV_stream):
    measurement_name = "visual_projection_field"
    ifclient = ifdb.create_ifclient()

    while True:
        (img, mask, frame_id) = high_level_vision_stream.get()
        logger.info(high_level_vision_stream.qsize())
        cropped_image = mask[projection.H_MARGIN:-projection.H_MARGIN, projection.W_MARGIN:-projection.W_MARGIN]
        projection_field = np.max(cropped_image, axis=0)

        if projection.SAVE_PROJECTION_FIELD:
            # Saving projection field data to InfluxDB to visualize with Grafana
            proj_field_vis = projection_field[0:-1:projection.DOWNGRADING_FACTOR]

            # take a timestamp for this measurement
            time = datetime.datetime.utcnow()

            # generating data to dump in db
            keys = [f'{ifdb.pad_to_n_digits(i)}' for i in range(len(proj_field_vis))]
            field_dict = dict(zip(keys, proj_field_vis))

            # format the data as a single measurement for influx
            body = [
                {
                    "measurement": measurement_name,
                    "time": time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    "fields": field_dict
                }
            ]

            ifclient.write_points(body)
