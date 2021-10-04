import cv2
import os
import time
import argparse
import random
import math
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_process.parse_frame import extract_image_and_label_from_frame, extract_frame_list
from waymo_process.schedule_frame import *
from waymo_process.partial_frame_postprocess import *
from waymo_process.waymo_test_utils import *

from scheduling_utils.GeneralObjectBuffer import GeneralObjectBuffer
from scheduling_utils.ObjectTubelet import ObjectTubelet
from scheduling_utils.TaskObject import TaskObject
from scheduling_utils.IoU import get_iou



'''
Requirement:
    1) Python 3.6 + Tensorflow 2.0 + waymo_open_dataset
'''


def test_single_file(model, input_file, output_path, use_random_border=True):
    '''
    The test function for a given input Waymo record.
    '''
    # print log information
    print("Testing: " + input_file)

    # Extract the whole segment, 200 frames within each segment (20s); list of frames
    # for each frame:
    # {
    #   'cameraName': {
    #                   'image',
    #                   'image_id' = 'timestamp' + '_' + 'cameraName',
    #                   'bbox_list' : [{'class', 'value', 'risk', 'object_id'}]
    #                  }
    # }
    start = time.time()
    frame_list = extract_frame_list(input_file, use_single_camera=True, load_one_frame=False)
    frame_count = len(frame_list)
    end = time.time()
    print("------------------------------------------------------------------------")
    print("Frame count: " + str(frame_count))
    print("File reading and parsing time: %f s" % (end - start))

    # # ---------------------------------------------Scheduling & Inference---------------------------------------------
    '''
    Each frame is scheduled and predicted in serialized order.
    '''
    # warm up run before testing
    for i in range(32, 128, 32):
        print(i)
        for j in range(32, 128, 32):
            warmup_image = np.random.randint(low=0, high=256, size=[1, i, j, 3])
            model.predict_on_batch(warmup_image)

    # scheduling
    object_buffer = GeneralObjectBuffer()
    finished_objects = []
    obj_id = 0

    # for deduplication
    iou_threshold = 0.7
    duplication_count = 0
    correct_duplication_count = 0
    object_pool = dict()
    tubelet_list = []
    global_tubelet_id = 0
    previous_frame_objects = list()

    # start time
    current_time = 0
    sampling_period = 10

    # main loop for inference
    for i, frame in enumerate(frame_list):
        print("----------------------------- Start frame: " + str(i) + "-----------------------------")
        # compute period end
        period_start = i * sampling_period
        period_end = (i + 1) * sampling_period

        # test and update current time
        if current_time < period_start:
            current_time = period_start

        # placeholder for current objects and previous objects
        current_frame_objects = list()
        duplication_count_for_current_frame = 0

        # extract all partial objects
        for camera_name in frame:
            # complete image
            image = frame[camera_name]['image']
            bbox_list = frame[camera_name]['bbox_list']
            image_id = frame[camera_name]['image_id']

            # original image size
            original_max_x = image.shape[1]
            original_max_y = image.shape[0]

            # iterate over every bbox
            for bbox in bbox_list:
                min_x, min_y, max_x, max_y = bbox['value']
                min_x = math.floor(min_x)
                min_y = math.floor(min_y)
                max_x = math.ceil(max_x)
                max_y = math.ceil(max_y)

                if use_random_border:
                    # random_border_len = random.randint(0, 9)
                    random_border_len = 10
                    min_x = max(0, min_x - random_border_len)
                    min_y = max(0, min_y - random_border_len)
                    max_x = min(original_max_x, max_x + random_border_len)
                    max_y = min(original_max_y, max_y + random_border_len)

                bbox_value = {'x1': min_x, 'y1': min_y, 'x2': max_x, 'y2': max_y}

                # extract partial image, [1, length, height], will be resized into multiplies of 32
                partial_image = image[min_y:max_y, min_x:max_x, :]
                preprocessed_partial_image, original_size, new_size = utils.image_preprocess(np.copy(partial_image))
                preprocessed_partial_image = preprocessed_partial_image[np.newaxis, ...]

                # formulate a new object, replace the criticality here
                criticality = 1
                partial_object = TaskObject(obj_id, image_id, preprocessed_partial_image,
                                            current_time, original_size, new_size, criticality,
                                            bbox=bbox_value, waymo_id=bbox['object_id'])

                # test duplication.
                object_pool[obj_id] = partial_object
                find_previous_object = False
                for old_object in previous_frame_objects:
                    iou = get_iou(old_object.bbox, partial_object.bbox)
                    if iou > iou_threshold:
                        find_previous_object = True
                        duplication_count_for_current_frame += 1
                        duplication_count += 1
                        partial_object.tubelet = old_object.tubelet

                        # check correct deduplication
                        if old_object.waymo_id == partial_object.waymo_id:
                            correct_duplication_count += 1

                        # only add to live tubelets; otherwise, check other tubelets or create a new one
                        if not old_object.tubelet.is_dead:
                            # add to the existing tubelet
                            old_object.tubelet.add_new_object(partial_object)

                            # if the tubelet is finished, directly save the predictions for the new object.
                            if old_object.tubelet.is_finished:
                                partial_object.is_finished = True
                                partial_object.finish_time = current_time

                                # copy predictions
                                partial_object.copy_predictions(old_object.tubelet.newest_object)

                            # only consider the first overlapped object
                            break
                        else:
                            continue

                # no duplication, push into buffer and create a new tubelet
                if not find_previous_object:
                    new_tubelet = ObjectTubelet(global_tubelet_id, partial_object)
                    partial_object.tubelet = new_tubelet
                    global_tubelet_id += 1
                    tubelet_list.append(new_tubelet)
                    object_buffer.push_new_object(partial_object)

                # add to current object list
                current_frame_objects.append(partial_object)

                # update object_id
                obj_id += 1

        # save all objects in current frame
        previous_frame_objects = current_frame_objects

        print("Number of objects in current frame: " + str(len(current_frame_objects)))
        print("Number of duplications in current frame: " + str(duplication_count_for_current_frame))

        # main loop of scheduling and execution
        while True:
            # stop the loop if current period is finished
            if current_time >= period_end:
                break

            current_time = object_buffer.execute_one_batch_with_tubelet(current_time, model)

            # stop the loop if no execution is performed
            if object_buffer.is_empty():
                break

        # break after one frame
        if i > 10:
            break

    # extract objects from all tubelets
    for tubelet in tubelet_list:
        for object in tubelet.object_list:
            if object.is_finished:
                finished_objects.append(object)

    # map the predictions to the global frame
    frame_predictions = dict()
    for object in finished_objects:
        image_id = object.image_id
        if image_id not in frame_predictions:
            frame_predictions[image_id] = []

        # map local predictions
        mapped_obj_predictions = map_partial_object_predictions_to_global(object)
        frame_predictions[image_id].extend(mapped_obj_predictions)

    # save prediction files
    segment_id = os.path.basename(input_file).split('.')[0]
    prediction_path = os.path.join(output_path, "predictions", segment_id)

    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    for image_id in frame_predictions:
        prediction_file = os.path.join(prediction_path, image_id + ".txt")
        save_frame_predictions(prediction_file, frame_predictions[image_id])

    # analyze results
    print("-------------------------------------------------------------")
    print("Total objects count: " + str(obj_id))
    print("Duplication count: " + str(duplication_count))
    print("Correct duplication count: " + str(correct_duplication_count))
    print("Finished objects: " + str(len(finished_objects)))

    return finished_objects


if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-GPU", type=str,
                        default="False",
                        help="Flag about wheter to use GPU for inference")
    args = parser.parse_args()

    if args.GPU == "True" or args.GPU == "true":
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---------------------------------------------------- Define model ------------------------------------------------
    # NOTE: the shape param does not include the batch size
    input_layer = tf.keras.layers.Input([None, None, 3])
    feature_maps = YOLOv3(input_layer)

    # decode bounding boxes
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(inputs=input_layer, outputs=bbox_tensors)
    utils.load_weights(model, "../model/yolov3.weights")

    # -----------------------------------------------Load Waymo data----------------------------------------------------
    '''
    Output is a numpy array of given size: [1, input_size, input_size, 3].
    '''
    input_path = "../data/Waymo/validation"
    output_path = "../results"
    input_files = extract_files(input_path)
    # random.shuffle(input_files)

    for input_file in input_files:
        # if "287009" not in input_file:
        #     continue
        finished_objects = test_single_file(model, input_file, output_path)
        break

        # break
