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
    for j in range(2000):
        img = raw_vision_stream.get()
        logger.info(raw_vision_stream.qsize())
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsvimg, segmentation.HSV_LOW, segmentation.HSV_HIGH)

        cv2.imshow("Raw", cv2.resize(img, (320, 240)))
        cv2.imshow("Processed", cv2.resize(mask, (320, 240)))
        high_level_vision_stream.put(mask)
        cv2.waitKey(1)

