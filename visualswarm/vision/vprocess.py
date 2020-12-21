"""
@author: mezdahun
@description: Processing low-level input to get High level input
"""
import logging
import cv2
import numpy as np

from visualswarm.contrib import segmentation

# using main logger
logger = logging.getLogger('visualswarm.app')


def nothing(x):
    pass

def high_level_vision(raw_vision_stream, high_level_vision_stream):
    """
    Process to process raw vision into high level vision and push it to a dedicated stream so that other behavioral
    processes can consume this stream
        Args:
            raw_vision_stream: multiprocessing.Queue type object to read raw visual input.
            high_level_vision_stream: multiprocessing.Queue type object to push high-level visual data.
        Returns:
            -shall not return-
    """
    target_hsv = segmentation.TARGET_HSV_COLOR
    hsv_low = segmentation.HSV_LOW
    hsv_high = segmentation.HSV_HIGH

    if segmentation.FIND_COLOR_INTERACTIVE:
        cv2.namedWindow("Segmentation Parameters")
        cv2.createTrackbar("R", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("G", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("B", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("H_range", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("SV_min", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("SV_max", "Segmentation Parameters", 0, 255, nothing)

    while True:
        img = raw_vision_stream.get()
        if segmentation.FIND_COLOR_INTERACTIVE:
            B = cv2.getTrackbarPos("B", "Segmentation Parameters")
            G = cv2.getTrackbarPos("G", "Segmentation Parameters")
            R = cv2.getTrackbarPos("R", "Segmentation Parameters")
            hue_range = cv2.getTrackbarPos("H_range", "Segmentation Parameters")
            sv_min = cv2.getTrackbarPos("SV_min", "Segmentation Parameters")
            sv_max = cv2.getTrackbarPos("SV_max", "Segmentation Parameters")
            target_hsv = cv2.cvtColor(np.uint8([[[B, G, R]]]), cv2.COLOR_BGR2HSV)
            hsv_low = np.uint8([target_hsv[0][0][0] - hue_range, sv_min, sv_min])
            hsv_high = np.uint8([target_hsv[0][0][0] + hue_range, sv_max, sv_max])

        # logger.info(raw_vision_stream.qsize())
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsvimg, hsv_low, hsv_high)

        cv2.imshow("Raw", cv2.resize(img, (160, 120)))
        cv2.imshow("Processed", cv2.resize(mask, (160, 129)))
        high_level_vision_stream.put(mask)
        cv2.waitKey(1)

