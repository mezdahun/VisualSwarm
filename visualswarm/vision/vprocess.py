"""
@author: mezdahun
@description: Processing low-level input to get High level input
"""
import datetime
from math import floor
import os
import shutil

import cv2
import numpy as np

import visualswarm.contrib.vision
from visualswarm import env
from visualswarm.monitoring import ifdb
from visualswarm.contrib import camera, vision, monitoring, simulation

from datetime import datetime

if vision.RECOGNITION_TYPE == "CNN":
    import importlib.util
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

from pprint import pformat

# using main logger
if not simulation.ENABLE_SIMULATION:
    import logging
    import os
    ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
    logger = logging.getLogger(f'VSWRM|{ROBOT_NAME}')
    logger.setLevel(monitoring.LOG_LEVEL)
else:
    logger = logging.getLogger('visualswarm.app_simulation')   # pragma: simulation no cover
from time import sleep


def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)

def nothing(x):
    pass

def get_latest_element(queue):
    """
    emptying a FIFO Queue object from multiprocessing package as there is no explicit way to do this.
        Args:
            queue2empty (multiprocessing.Queue): queue object to be emptied
        Returns:
            status: True if successful
    """
    val = None
    while not queue.empty():
        try:
            val = queue.get_nowait()
        except Empty:
            return val
    return val

def high_level_vision(raw_vision_stream, high_level_vision_stream, visualization_stream=None,
                      target_config_stream=None):
    """
    Process to process raw vision into high level vision and push it to a dedicated stream so that other behavioral
    processes can consume this stream
        Args:
            raw_vision_stream (multiprocessing.Queue): stream object to read raw visual input.
            high_level_vision_stream (multiprocessing.Queue): stream to push high-level visual data.
            visualization_stream (multiprocessing.Queue): stream to visualize raw vs processed vision, and to tune
                parameters interactively
            target_config_stream (multiprocessing.Queue): stream to transmit configuration parameters if interactive
                configuration is turned on.
        Returns:
            -shall not return-
    """
    try:
        if vision.RECOGNITION_TYPE == "Color":
            hsv_low = visualswarm.contrib.vision.HSV_LOW
            hsv_high = visualswarm.contrib.vision.HSV_HIGH

        while True:

            if vision.RECOGNITION_TYPE == "Color":

                (img, frame_id, capture_timestamp) = raw_vision_stream.get()
                # logger.info(raw_vision_stream.qsize())
                if vision.FIND_COLOR_INTERACTIVE:
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
                blurred = cv2.GaussianBlur(mask, (
                    visualswarm.contrib.vision.GAUSSIAN_KERNEL_WIDTH, visualswarm.contrib.vision.GAUSSIAN_KERNEL_WIDTH), 0)
                blurred = cv2.medianBlur(blurred, visualswarm.contrib.vision.MEDIAN_BLUR_WIDTH)

                # Find contours
                conts, h = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

                # Selecting appropriate contours
                fconts = [cnt for cnt in conts if cv2.contourArea(cnt) >= visualswarm.contrib.vision.MIN_BLOB_AREA]

                # Creating convex hull from selected contours
                hull_list = []
                for i in range(len(fconts)):
                    hull = cv2.convexHull(fconts[i])
                    hull_list.append(hull)

                # visualize contours and convex hull on the original image and the area on the new mask
                cv2.drawContours(img, fconts, -1, vision.RAW_CONTOUR_COLOR, vision.RAW_CONTOUR_WIDTH)
                cv2.drawContours(img, hull_list, -1, vision.CONVEX_CONTOUR_COLOR, vision.CONVEX_CONTOUR_WIDTH)
                cv2.drawContours(blurred, hull_list, -1, (255, 255, 255), -1)

            # Forwarding result to VPF extraction
            high_level_vision_stream.put((img, blurred, frame_id, capture_timestamp))

            # Forwarding result for visualization if requested
            if visualization_stream is not None:
                visualization_stream.put((img, blurred, frame_id))

            # To test infinite loops
            if env.EXIT_CONDITION:
                break

    except KeyboardInterrupt:
        pass


def high_level_vision_CNN(raw_vision_stream, high_level_vision_stream, visualization_stream=None,
                      target_config_stream=None):
    """
    Process to process raw vision into high level vision and push it to a dedicated stream so that other behavioral
    processes can consume this stream
        Args:
            raw_vision_stream (multiprocessing.Queue): stream object to read raw visual input.
            high_level_vision_stream (multiprocessing.Queue): stream to push high-level visual data.
            visualization_stream (multiprocessing.Queue): stream to visualize raw vs processed vision, and to tune
                parameters interactively
            target_config_stream (multiprocessing.Queue): stream to transmit configuration parameters if interactive
                configuration is turned on.
        Returns:
            -shall not return-
    """
    import logging
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    from picamera.exc import PiCameraValueError
    import cv2

    import time
    from datetime import datetime

    from visualswarm.contrib import camera, logparams

    # using main logger
    logger = logging.getLogger('visualswarm.app')
    bcolors = logparams.BColors

    logger.info('Loading tensorflow model...')
    MODEL_NAME = '/home/pi/VisualSwarm/CNNtools/data/tflite_model/edgetpu'
    GRAPH_NAME = 'fixedlense_fullinteger_edgetpu.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    USE_TPU = True
    INTQUANT = True
    # it takes a little longer on the first run and then runs at normal speed.
    import random
    import glob

    if USE_TPU:
        pkg = importlib.util.find_spec('tflite_runtime')
        if not pkg:
            from tensorflow.lite.python.interpreter import load_delegate
        else:
            from tflite_runtime.interpreter import load_delegate

    min_conf_threshold = 0.25

    resW, resH = camera.RESOLUTION
    imW, imH = int(resW), int(resH)

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if USE_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    logger.info('Model loaded!')

    if monitoring.SAVE_CNN_TRAINING_DATA:
        training_data_folder = os.path.join(monitoring.SAVED_VIDEO_FOLDER, 'training_data')
        if os.path.isdir(training_data_folder):
            shutil.rmtree(training_data_folder)
        os.makedirs(training_data_folder, exist_ok=True)

    try:
        try:
            try:
                picam = PiCamera()
                picam.resolution = camera.RESOLUTION
                picam.framerate = camera.FRAMERATE
                logger.debug(f'\n{bcolors.OKBLUE}--Camera Params--{bcolors.ENDC}\n'
                             f'{bcolors.OKBLUE}Resolution:{bcolors.ENDC} {camera.RESOLUTION} px\n'
                             f'{bcolors.OKBLUE}Frame Rate:{bcolors.ENDC} {camera.FRAMERATE} fps')

                # Generates a 3D RGB array and stores it in rawCapture
                raw_capture = PiRGBArray(picam, size=camera.RESOLUTION)

                # Wait a certain number of seconds to allow the camera time to warmup
                logger.info('Waiting 8 secs for camera warmup!')
                time.sleep(8)
                frame_id = 0
                for frame in picam.capture_continuous(raw_capture,
                                                      format=camera.CAPTURE_FORMAT,
                                                      use_video_port=camera.USE_VIDEO_PORT):
                    # Grab the raw NumPy array representing the image
                    if camera.FLIP_CAMERA:
                        img = cv2.flip(frame.array, -1)
                    else:
                        img = frame.array

                    # Clear the raw capture stream in preparation for the next frame
                    raw_capture.truncate(0)

                    # Adding time of capture for delay measurement
                    capture_timestamp = datetime.utcnow()

                    # Collecting training data in a predefined freq
                    if frame_id == 0:
                        CNN_TD_last_collect = capture_timestamp

                    # clear vision stream if polluted to avoid delay
                    t0 = capture_timestamp

                    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (width, height))
                    input_data = np.expand_dims(frame_resized, 0).astype('float32')

                    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                    if floating_model:
                        input_data = (np.float32(input_data) - input_mean) / input_std
                    if INTQUANT:
                        input_data = input_data.astype('uint8')

                    t1 = datetime.utcnow()
                    logger.info(f'preprocess time {(t1 - t0).total_seconds()}')
                    # Perform the actual detection by running the model with the image as input
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()

                    # Bounding box coordinates of detected objects
                    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
                    # Class index of detected objects
                    # classes = interpreter.get_tensor(output_details[1]['index'])[0]
                    # Confidence of detected objects
                    scores = interpreter.get_tensor(output_details[2]['index'])[0]

                    # Dequantize if input and output is int quantized
                    if INTQUANT:
                        scale, zero_point = output_details[0]['quantization']
                        boxes = scale * (boxes - zero_point)

                        # scale, zero_point = output_details[1]['quantization']
                        # classes = scale * (classes - zero_point)

                        scale, zero_point = output_details[2]['quantization']
                        scores = scale * (scores - zero_point)

                    t2 = datetime.utcnow()
                    delta = (t2 - t1).total_seconds()
                    logger.debug(f"Inference time: {delta}, rate={1 / delta}")  #

                    blurred = np.zeros([img.shape[0], img.shape[1]])

                    for i in range(len(boxes)):
                        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                            # if scores[i] == np.max(scores):
                            # Get bounding box coordinates and draw box
                            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                            ymin = int(max(1, (boxes[i, 0] * imH)))
                            xmin = int(max(1, (boxes[i, 1] * imW)))
                            ymax = int(min(imH, (boxes[i, 2] * imH)))
                            xmax = int(min(imW, (boxes[i, 3] * imW)))

                            blurred[ymin:ymax, xmin:xmax] = 255
                            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                            img = cv2.putText(img, f'score={scores[i]:.2f}', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, (255, 0, 0), 2, cv2.LINE_AA)

                    t3 = datetime.utcnow()
                    logger.debug(f"Postprocess time: {(t3 - t1).total_seconds()}")

                    # Forwarding result to VPF extraction
                    logger.debug(f'Queue length{raw_vision_stream.qsize()}')
                    high_level_vision_stream.put((img, blurred, frame_id, capture_timestamp))
                    t4 = datetime.utcnow()
                    logger.debug(f'Transferring time: {(t4 - t3).total_seconds()}')

                    # Collecting training data for CNN fine tune if requested
                    if monitoring.SAVE_CNN_TRAINING_DATA:
                        if (capture_timestamp - CNN_TD_last_collect).total_seconds() > 1/monitoring.CNN_TRAINING_DATA_FREQ:
                            frame_name = f'{EXP_ID}_{ROBOT_NAME}_CNNTD_frame{frame_id}.png'
                            frame_path = os.path.join(training_data_folder, frame_name)
                            cv2.imwrite(frame_path, frame_rgb)

                    # Forwarding result for visualization if requested
                    if visualization_stream is not None:
                        visualization_stream.put((img, blurred, frame_id))

                    # To test infinite loops
                    if env.EXIT_CONDITION:
                        break

                    t5 = datetime.utcnow()
                    logger.info(f'total vision_rate: {1 / (t5 - t0).total_seconds()}')

                    frame_id += 1
            except KeyboardInterrupt:
                try:
                    pass
                except PiCameraValueError:
                    pass
        except PiCameraValueError:
            pass

    except KeyboardInterrupt:
        pass


def visualizer(visualization_stream, target_config_stream=None):
    """
    Process to Visualize Raw and Processed camera streams via a visualization stream. It is also used to tune parameters
    interactively, in this case a configuration stream is also used to fetch interactively given parameters from the
    user.
        Args:
            visualization_stream (multiprocessing.Queue): stream to visualize raw vs processed vision, and to tune
                parameters interactively
            target_config_stream (multiprocessing.Queue): stream to transmit segmentation parameters between
                interactive tuning input window and the visualization_stream
        Returns:
            -shall not return-
    """
    try:
        if visualization_stream is not None:
            if vision.FIND_COLOR_INTERACTIVE:
                cv2.namedWindow("Segmentation Parameters")
                cv2.createTrackbar("R", "Segmentation Parameters", visualswarm.contrib.vision.TARGET_RGB_COLOR[0], 255,
                                   nothing)
                cv2.createTrackbar("G", "Segmentation Parameters", visualswarm.contrib.vision.TARGET_RGB_COLOR[1], 255,
                                   nothing)
                cv2.createTrackbar("B", "Segmentation Parameters", visualswarm.contrib.vision.TARGET_RGB_COLOR[2], 255,
                                   nothing)
                cv2.createTrackbar("H_range", "Segmentation Parameters", visualswarm.contrib.vision.HSV_HUE_RANGE, 255,
                                   nothing)
                cv2.createTrackbar("SV_min", "Segmentation Parameters", visualswarm.contrib.vision.SV_MINIMUM,
                                   255, nothing)
                cv2.createTrackbar("SV_max", "Segmentation Parameters", visualswarm.contrib.vision.SV_MAXIMUM,
                                   255, nothing)
                color_sample = np.zeros((200, 200, 3), np.uint8)

            if monitoring.SAVE_VISION_VIDEO:
                ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
                EXP_ID = os.getenv('EXP_ID', 'expXXXXXX')
                video_timestamp = datetime.now().strftime("%d-%m-%y-%H%M%S")
                os.makedirs(monitoring.SAVED_VIDEO_FOLDER, exist_ok=True)
                video_name = os.path.join(monitoring.SAVED_VIDEO_FOLDER, f'{video_timestamp}_{EXP_ID}_{ROBOT_NAME}.mp4')
                writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), camera.FRAMERATE,
                                         camera.RESOLUTION, isColor=True)

            while True:
                # trick to release video on time upon process destruction via SSH. When using SSH CtrlC can not be
                # achieved and is simulated with SIGINT. But because of this the video is not finalized in the exception
                # block. for desired behavior do
                # $ touch release.txt && sleep 2 && rm -rf release.txt
                # then kill the process
                if os.path.isfile('/home/pi/VisualSwarm/release.txt'):
                    writer.release()
                    return

                (img, mask, frame_id) = visualization_stream.get()
                if vision.FIND_COLOR_INTERACTIVE:
                    if target_config_stream is not None:
                        B = cv2.getTrackbarPos("B", "Segmentation Parameters")
                        G = cv2.getTrackbarPos("G", "Segmentation Parameters")
                        R = cv2.getTrackbarPos("R", "Segmentation Parameters")
                        color_sample[:] = [B, G, R]
                        HSV_HUE_RANGE = cv2.getTrackbarPos("H_range", "Segmentation Parameters")
                        SV_MINIMUM = cv2.getTrackbarPos("SV_min", "Segmentation Parameters")
                        SV_MAXIMUM = cv2.getTrackbarPos("SV_max", "Segmentation Parameters")
                        target_config_stream.put((R, B, G, HSV_HUE_RANGE, SV_MINIMUM, SV_MAXIMUM))

                if vision.SHOW_VISION_STREAMS:
                    vis_width = floor(camera.RESOLUTION[0] / vision.VIS_DOWNSAMPLE_FACTOR)
                    vis_height = floor(camera.RESOLUTION[1] / vision.VIS_DOWNSAMPLE_FACTOR)
                    cv2.imshow("Object Contours", cv2.resize(img, (vis_width, vis_height)))
                    cv2.imshow("Final Area", cv2.resize(mask, (vis_width, vis_height)))
                    if vision.FIND_COLOR_INTERACTIVE:
                        cv2.imshow("Segmentation Parameters", color_sample)
                    cv2.waitKey(1)

                if monitoring.SAVE_VISION_VIDEO:
                    mask_to_write = cv2.resize(img, camera.RESOLUTION)
                    writer.write(mask_to_write)

                # To test infinite loops
                if env.EXIT_CONDITION:
                    break
        else:
            logger.info('Visualization stream is None, visualization process returns!')
    except KeyboardInterrupt:
        if monitoring.SAVE_VISION_VIDEO:
            writer.release()
    except:
        writer.release()


def VPF_extraction(high_level_vision_stream, VPF_stream):
    """
    Process to extract final visual projection field from high level visual input.
        Args:
            high_level_vision_stream (multiprocessing.Queue): Stream object to get processed visual information
            VPF_stream (multiprocessing.Queue): stream to push final visual projection field
        Returns:
            -shall not return-
    """
    try:
        if not simulation.ENABLE_SIMULATION:
            measurement_name = "visual_projection_field"
            ifclient = ifdb.create_ifclient()

        while True:
            (img, mask, frame_id, capture_timestamp) = high_level_vision_stream.get()
            # logger.info(high_level_vision_stream.qsize())
            cropped_image = mask[visualswarm.contrib.vision.H_MARGIN:-visualswarm.contrib.vision.H_MARGIN,
                                 visualswarm.contrib.vision.W_MARGIN:-visualswarm.contrib.vision.W_MARGIN]
            projection_field = np.max(cropped_image, axis=0)
            projection_field = projection_field / 255

            if monitoring.SAVE_PROJECTION_FIELD and not simulation.ENABLE_SIMULATION:
                # Saving projection field data to InfluxDB to visualize with Grafana
                proj_field_vis = projection_field[0:-1:monitoring.DOWNGRADING_FACTOR]

                # take a timestamp for this measurement
                time = datetime.datetime.utcnow()

                # generating data to dump in db
                keys = [f'{ifdb.pad_to_n_digits(i)}' for i in range(len(proj_field_vis))]
                field_dict = dict(zip(keys, proj_field_vis))

                # format the data as a single measurement for influx
                body = [
                    {
                        "measurement": measurement_name,
                        "time": time,
                        "fields": field_dict
                    }
                ]

                ifclient.write_points(body, time_precision='ms')

            VPF_stream.put((projection_field, capture_timestamp))

            # To test infinite loops
            if env.EXIT_CONDITION:
                break

    except KeyboardInterrupt:
        pass
