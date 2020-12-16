"""
@author: mezdahun
@description: Processing low-level input to get High level input
"""

import cv2


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
        print(type(img))
        cv2.imshow("Frame", img)
        cv2.waitKey(1)