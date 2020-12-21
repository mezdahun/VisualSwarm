"""
@author: mezdahun
@description: Processing low-level input to get High level input
"""
import logging
import cv2
# import numpy as np

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
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("B", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("G", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("R", "Trackbars", 0, 255, nothing)

    for j in range(2000):
        B = cv2.getTrackbarPos("B", "Trackbars")
        G = cv2.getTrackbarPos("G", "Trackbars")
        R = cv2.getTrackbarPos("R", "Trackbars")
        img = raw_vision_stream.get()
        TARGET_HSV_COLOR = cvtColor(uint8([[[B, G, R]]]), COLOR_BGR2HSV)
        HSV_LOW = uint8([TARGET_HSV_COLOR[0][0][0] - HSV_HUE_RANGE, SV_MINIMUM, SV_MINIMUM])
        HSV_HIGH = uint8([TARGET_HSV_COLOR[0][0][0] + HSV_HUE_RANGE, SV_MAXIMUM, SV_MAXIMUM])

        logger.info(raw_vision_stream.qsize())
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsvimg, HSV_LOW, HSV_HIGH)

        cv2.imshow("Raw", cv2.resize(img, (320, 240)))
        cv2.imshow("Processed", cv2.resize(mask, (320, 240)))
        high_level_vision_stream.put(mask)
        cv2.waitKey(1)

