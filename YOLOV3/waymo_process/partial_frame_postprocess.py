import numpy as np
import core.utils as utils

'''
Postprocessing for partial frames. Convert the partial predictions to original frames.
'''


def postprocess_box_one_batch(pred_bbox, frame_meta):
    '''
    Post process function for partial/complete frame predictions. Attach necessary meta info to the prediction.
    Map the bbox from new sizes back to original sizes.
    TODO: Currently only support one frame per batch.
    '''
    original_image_size = frame_meta['original_size']
    new_size = frame_meta['new_size']

    # adjust the predicted bounding boxes at current frames
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, new_size, 0.3)

    # non maximum supppression
    bboxes = np.array(utils.nms(bboxes, 0.45, method='nms'))

    # ensemble prediction item
    prediction_bbox_item = {'image_id': frame_meta['image_id'],
                            'camera_name': frame_meta['camera_name'],
                            'bboxes': bboxes}

    if 'partial_frame_offset' in frame_meta:
        prediction_bbox_item['partial_frame_offset'] = frame_meta['partial_frame_offset']

    return prediction_bbox_item


def map_partial_bbox_to_global(bboxes, partial_frame_offset):
    '''
    Map partial box predictions back to its original frame.
    '''
    mapped_bboxes = np.copy(bboxes)

    if len(bboxes) == 0:
        return mapped_bboxes
    else:
        mapped_bboxes[:, [0, 2]] += partial_frame_offset[0]
        mapped_bboxes[:, [1, 3]] += partial_frame_offset[1]
        return mapped_bboxes


def map_partial_object_predictions_to_global(object):
    '''
    Map the local predictions of each object to its original frame.
    :param object:
    :return:
    '''
    global_bboxes = np.copy(object.predictions)

    x_offset = object.bbox['x1']
    y_offset = object.bbox['y1']

    if len(global_bboxes) == 0:
        return []
    else:
        global_bboxes[:, [0, 2]] += x_offset
        global_bboxes[:, [1, 3]] += y_offset
        return list(global_bboxes)


def merge_partial_pred_bbox(pred_bbox_list):
    '''
    Merge and map all partial pred bbox.
    Output: dict(), key is the camera name, value is their corresponding bbox prediction list.
    bbox: [min_x, min_y, max_x, max_y, conf, class]
    '''
    frame_prediction = {}

    for prediction_bbox_item in pred_bbox_list:
        image_id = prediction_bbox_item['image_id']
        bboxes = prediction_bbox_item['bboxes']
        partial_frame_offset = prediction_bbox_item['partial_frame_offset']

        # map partial bboxes to global bboxes
        mapped_bboxes = map_partial_bbox_to_global(bboxes, partial_frame_offset)

        if image_id not in frame_prediction:
            frame_prediction[image_id] = []

        frame_prediction[image_id].extend(mapped_bboxes)

    return frame_prediction


def extract_full_pred_bbox(pred_bbox_list):
    '''
    Used by serialized full frame scheduling. Convert bbox list to required format.
    '''
    frame_prediction = {}

    for prediction_bbox_item in pred_bbox_list:
        image_id = prediction_bbox_item['image_id']
        bboxes = prediction_bbox_item['bboxes']

        if image_id not in frame_prediction:
            frame_prediction[image_id] = []

        frame_prediction[image_id].extend(bboxes)

    return frame_prediction
