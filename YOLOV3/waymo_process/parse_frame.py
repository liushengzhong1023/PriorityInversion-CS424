import numpy as np
import math
import cv2
import os
import time
import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import core.utils as utils

'''
Object class mapping:
      enum Type {
        TYPE_UNKNOWN = 0;
        TYPE_VEHICLE = 1;
        TYPE_PEDESTRIAN = 2;
        TYPE_SIGN = 3;
        TYPE_CYCLIST = 4;
      }
'''

waymo_to_coco = {0: 10, 1: 2, 2: 0, 3: 11, 4: 1}  # from waymo to coco


def convert_bounding_box(box):
    '''
    Convert from [center_x, center_y, length, width] --> [x_min, y_min, x_max, y_max]
    '''
    return [box.center_x - 0.5 * box.length, box.center_y - 0.5 * box.width,
            box.center_x + 0.5 * box.length, box.center_y + 0.5 * box.width]


def parse_image_from_buffer(image_buffer):
    '''
    Serialized image buffer --> numpy array
    '''
    # print(image_buffer)
    image = np.fromstring(image_buffer, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def extract_image_and_label_from_frame(frame, label_path, use_single_camera=False):
    '''
    Extract the 2d image, and its corresponding labels from the given frame.
    NOTE: 5 cameras deployed (front, front left, front right, side left, side right); JPEG image for each camera.
    output: Dict of 5 dict, where the first level key is image name;
            The second level dict['image']=image, dict['bbox_list']=list of bounding boxes
    '''
    parsed_frame = dict()
    frame_id = frame.timestamp_micros

    # process front camera only
    if use_single_camera:
        target_camera_names = {open_dataset.CameraName.FRONT}
    else:
        target_camera_names = {open_dataset.CameraName.FRONT,
                               open_dataset.CameraName.FRONT_LEFT,
                               open_dataset.CameraName.FRONT_RIGHT,
                               open_dataset.CameraName.SIDE_LEFT,
                               open_dataset.CameraName.SIDE_RIGHT}

    # extract the image for each camera
    for image in frame.images:
        image_camera_name = image.name

        # Skip cameras with unknown camera names
        if image_camera_name not in target_camera_names:
            continue

        # Compute camera name
        # image_id = str(frame.context.name) + '_' + str(frame_count) + '_' + str(image_camera_name)
        image_id = str(frame_id) + '_' + str(image_camera_name)

        if image_camera_name not in parsed_frame:
            parsed_frame[image_camera_name] = dict()
        parsed_frame[image_camera_name]['image'] = parse_image_from_buffer(image.image)
        parsed_frame[image_camera_name]['image_id'] = image_id

    # extract the bounding boxes for each camera
    for camera_label in frame.camera_labels:
        label_camera_name = camera_label.name
        if label_camera_name not in parsed_frame:
            continue
        else:
            # Create list of bounding box
            if 'bbox_list' not in parsed_frame[label_camera_name]:
                parsed_frame[label_camera_name]['bbox_list'] = []

            # Extract each bounding box within each camera_label
            for bbox_label in camera_label.labels:
                bbox = convert_bounding_box(bbox_label.box)
                bbox_class = int(bbox_label.type)
                object_id = bbox_label.id

                # get object risk (compute your own risk here)
                risk = 0

                parsed_frame[label_camera_name]['bbox_list'].append({'class': bbox_class,
                                                                     'value': bbox,
                                                                     'risk': risk,
                                                                     'object_id': object_id})

    # save the labels
    for camera_name in parsed_frame:
        output_file = os.path.join(label_path, parsed_frame[camera_name]['image_id'] + '.txt')
        bbox_list = parsed_frame[camera_name]['bbox_list']

        with open(output_file, 'w') as f:
            for bbox in bbox_list:
                bbox_coco_class = waymo_to_coco[int(bbox['class'])]
                x1, y1, x2, y2 = bbox['value']
                f.write(' '.join([str(e) for e in [bbox_coco_class, x1, y1, x2, y2]]) + '\n')


    return parsed_frame


def extract_frame_list(input_file, use_single_camera=False, load_one_frame=False):
    '''
    Extract frame list from the given video file.
    NOTE: data.numpy() requires TF eager execution. The .numpy() method explicitly converts a Tensor to a numpy array.
    '''
    label_path = "../results/labels"
    video_segment = tf.data.TFRecordDataset(input_file, compression_type='')
    segmend_id = os.path.basename(input_file).split('.')[0]
    label_path = os.path.join(label_path, segmend_id)

    if not os.path.exists(label_path):
        os.mkdir(label_path)

    frame_list = []
    frame_count = 0
    for data in video_segment:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        parsed_frame = extract_image_and_label_from_frame(frame, label_path, use_single_camera)

        frame_list.append(parsed_frame)
        frame_count += 1

        if load_one_frame:
            break

    return frame_list
