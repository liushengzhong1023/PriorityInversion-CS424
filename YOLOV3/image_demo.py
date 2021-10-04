#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:07:27
#   Description :
#
# ================================================================

import cv2
import time
import os
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_size = 416
image_path = "./docs/kite.jpg"

# input_layer = tf.keras.layers.Input([input_size, input_size, 3])
input_layer = tf.keras.layers.Input([1120, 1920, 3])
feature_maps = YOLOv3(input_layer)

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

# image_data, old_image_size, new_image_size = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
image_data, old_image_size, new_image_size = utils.image_preprocess(np.copy(original_image), [1120, 1920])
image_data = image_data[np.newaxis, ...].astype(np.float32)

image_data = np.tile(image_data, [1, 1, 1, 1])

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "/home/sl29/rvms/model/yolov3.weights")
# model.summary()

for i in range(10):
    pred_bbox = model.predict_on_batch(image_data)

start = time.time()

for i in range(100):
    pred_bbox = model.predict_on_batch(image_data)
    # pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    # pred_bbox = tf.concat(pred_bbox, axis=0)
    # bboxes = utils.postprocess_boxes(pred_bbox, old_image_size, new_image_size, 0.3)
    # bboxes = utils.nms(bboxes, 0.45, method='nms')
    # print(len(pred_bbox))
    # for item in pred_bbox:
    #     print(np.shape(item))
    # print()
    #
    # for i, object in enumerate(image_data):
    #     local_pred_bbox = [x[i] for x in pred_bbox]
    #     for item in local_pred_bbox:
    #         print(np.shape(item))
    #     print()
    # for bbox in bboxes:
    #     print(bbox)

end = time.time()
print("------------------------------------------------------------------------")
print("Inference time: %f s" % ((end - start) / 100.))

# image = utils.draw_bbox(original_image, bboxes)
# image = Image.fromarray(image)
# image.show()
