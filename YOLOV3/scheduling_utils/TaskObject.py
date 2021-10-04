import time
import tensorflow as tf
import numpy as np
import core.utils as utils


class TaskObject:
    def __init__(self, obj_id, image_id, image, arrival_time, original_size, new_size,
                 criticality, bbox=None, waymo_id=None):
        '''
        Notes:
            1) obj_id is a local assigned, not the global object id.
            2) bbox: {'x1': min_x, 'y1': min_y, 'x2': max_x, 'y2': max_y}
        '''
        # meta information for matching back to global frame
        self.image_id = image_id

        self.obj_id = obj_id
        self.criticality = criticality
        self.image = image

        # size before and after resize, [height (y), width (x)]
        self.original_size = original_size
        self.new_size = new_size

        # time related
        self.is_finished = False
        self.is_overdue = False
        self.arrival_time = arrival_time
        self.finish_time = 0
        self.deadline = self.arrival_time + 10

        # prediction result
        self.predictions = None

        # followings are used for deduplication
        self.bbox = bbox
        self.tubelet = None
        self.waymo_id = waymo_id


    def object_inference(self, current_time, model):
        '''
        Run inference on current object.
        returned value of predict_on_batch:
            * input: [batch, width, height, 3 (rgb)]
            * output: list of 3 ndarray, related to 3 scales (downsampling rate: 8, 16, 32).
                      for each scale, [batch_size, x, y, 3, (4 + 1 + 80)], x, y are downsampled dimensions;
                      3 boxes at each scale for each pixel; 4 bounding box offsets, 1 objectness predictoin;
                      80 class predictions in COCO.
            * Assume one image, we have x * y * 3 boxes predicted for each scale.
        :return:
        '''
        # print("Executing object id: " + str(self.obj_id))
        start = time.time()
        pred_bbox = model.predict_on_batch(self.image)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        # count time
        end = time.time()
        duration = end - start

        # save the results
        self.predictions = pred_bbox
        current_time += duration

        if current_time <= self.deadline:
            self.is_finished = True
        else:
            self.is_finished = False

        return current_time, pred_bbox


    def save_inference_result(self, pred_bbox, finish_time):
        '''
        After the inference on batch, save the predicted bbox into corresponding objects.
        :param pred_box:
        :param finish_time:
        :return:
        '''
        # shape: [boxes, 85]
        local_pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        local_pred_bbox = tf.concat(local_pred_bbox, axis=0)

        # post process on boxes and NMS, the boxes are mapped back to the original partial frame size
        # print(len(local_pred_bbox))
        bboxes = utils.postprocess_boxes(local_pred_bbox, self.original_size, self.new_size, 0.3)
        # print(len(bboxes))
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        # print(len(bboxes))
        # print()

        # save results
        self.predictions = bboxes

        # update state, only save the results if the due is not passed
        if finish_time <= self.deadline:
            self.is_finished = True
            self.finish_time = finish_time

            # update tubelet status
            self.tubelet.set_finish_time(finish_time, self.predictions, self)
        else:
            self.is_finished = False
            self.is_overdue = True

            # update tubelet status
            self.tubelet.set_death()


    def copy_predictions(self, agent_object):
        '''
        Copy predictions from the agent object to current object.
        :param agent_object:
        :return:
        '''
        predictions = []

        x_ratio = float(self.original_size[1]) / agent_object.original_size[1]
        y_ratio = float(self.original_size[0] / agent_object.original_size[0])

        # size mapping on the predictions, bbox: [x_min, y_min, x_max, y_max, objectness, coco_class]
        # All coordinates are corresponding to the *local* positions at original partial frames.
        for bbox in agent_object.predictions:
            x_min = min(bbox[0] * x_ratio, self.new_size[1])
            y_min = min(bbox[1] * y_ratio, self.new_size[0])
            x_max = min(bbox[2] * x_ratio, self.new_size[1])
            y_max = min(bbox[3] * y_ratio, self.new_size[0])

            new_bbox = [x_min, y_min, x_max, y_max, bbox[4], bbox[5]]
            predictions.append(new_bbox)

        self.predictions = predictions


