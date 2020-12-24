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

    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((10, 10))

    if segmentation.FIND_COLOR_INTERACTIVE:
        cv2.namedWindow("Segmentation Parameters")
        cv2.createTrackbar("R", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("G", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("B", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("H_range", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("SV_min", "Segmentation Parameters", 0, 255, nothing)
        cv2.createTrackbar("SV_max", "Segmentation Parameters", 0, 255, nothing)
        color_sample = np.zeros((200, 200, 3), np.uint8)

    while True:
        img = raw_vision_stream.get()
        if segmentation.FIND_COLOR_INTERACTIVE:
            B = cv2.getTrackbarPos("B", "Segmentation Parameters")
            G = cv2.getTrackbarPos("G", "Segmentation Parameters")
            R = cv2.getTrackbarPos("R", "Segmentation Parameters")
            color_sample[:] = [B, G, R]
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

        # Gaussian blur
        blurred = cv2.GaussianBlur(mask, (15, 15), 0)

        # sobelX = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        # sobelY = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
        # sobel = np.hypot(sobelX, sobelY)
        # sobel[sobel > 255] = 255;  # Some values seem to go above 255. However RGB channels has to be within 0-255

        # maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        # maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
        # maskFinal = maskClose
        conts, h = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(blurred, conts, 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

        # for i in range(len(conts)):
        #     x, y, w, h = cv2.boundingRect(conts[i])
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if segmentation.SHOW_VISION_STREAMS or segmentation.FIND_COLOR_INTERACTIVE:
            # cv2.imshow("Raw", cv2.resize(median, (160, 120)))
            cv2.imshow("Processed", cv2.resize(blurred, (160, 129)))
        if segmentation.FIND_COLOR_INTERACTIVE:
            cv2.imshow("Segmentation Parameters", color_sample)

        # # Cleaning the segmented image
        # maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        # maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        high_level_vision_stream.put(mask)
        cv2.waitKey(1)

