import os
import json
import core.utils as utils
from PIL import Image
from pathlib import Path
from waymo_open_dataset import dataset_pb2 as open_dataset


def show_frame_predictions(frame_list, extracted_frame, frame_pred_bbox):
    '''
    Plot function to show the frame predictions
    '''
    # show the image and bounding boxes
    if len(frame_list) == 1:
        for camera_name in extracted_frame:
            image = extracted_frame[camera_name]['image']

            # add bounding boxes if exist
            if camera_name in frame_pred_bbox:
                pred_bboxes = frame_pred_bbox[camera_name]
                image = utils.draw_bbox(image, pred_bboxes)
            image = Image.fromarray(image)

            # save file
            file_path = "/home/sl29/svms/results/figures"
            file_name = os.path.join(file_path, open_dataset.CameraName.Name.Name(camera_name) + ".jpg")
            image.save(file_name)


def extract_files(input_path):
    '''
    Input validation files within the given directory.
    '''
    input_files = []
    file_names = os.listdir(input_path)

    for file in file_names:
        if "_with_camera_labels" in file:
            input_files.append(os.path.join(input_path, file))

    return input_files


def get_prediction_file_name(output_path, scheduling_policy, with_random_boarder=False):
    '''
    Parse the input file to get the corresponding output file name.
    '''
    if with_random_boarder:
        prediction_output_file = os.path.join(output_path,
                                              "YOLOv3.prediction." + scheduling_policy + ".with_boarder.json")
    else:
        prediction_output_file = os.path.join(output_path,
                                              "YOLOv3.prediction." + scheduling_policy + ".no_boarder.json")

    return prediction_output_file


def get_time_file_name(output_path, scheduling_policy, use_GPU=False, threads_num=None):
    '''
    Parse the input file to get the corresponding time output file name.
    '''
    if use_GPU:
        time_output_file = os.path.join(output_path, "YOLOv3.time." + scheduling_policy + ".GPU.json")
    else:
        if not threads_num:
            time_output_file = os.path.join(output_path, "YOLOv3.time." + scheduling_policy + ".CPU.json")
        else:
            time_output_file = os.path.join(output_path,
                                            "YOLOv3.time." + scheduling_policy + ".CPU_" + str(threads_num) + ".json")

    return time_output_file


def get_frame_output(frame_pred_bbox):
    '''
    Change the output format of frame pred bbox for output.
    '''
    frame_output = dict()

    for image_id in frame_pred_bbox:
        frame_output[image_id] = []

        for bbox in frame_pred_bbox[image_id]:
            min_x = bbox[0]
            min_y = bbox[1]
            max_x = bbox[2]
            max_y = bbox[3]
            score = bbox[4]
            label = int(bbox[5])

            frame_output[image_id].append([min_x, min_y, max_x, max_y, score, score, label])

    return frame_output


def merge_frame_predictions(segment_predictions, frame_predictions):
    '''
    Merge frame output into predictions.
    '''
    for image_id in frame_predictions:
        segment_predictions[image_id] = frame_predictions[image_id]

    return segment_predictions


def merge_segment_predictions(predictions, input_file, segment_predictions):
    '''
    Update the segment output into the output.
    '''
    input_base = os.path.basename(input_file)
    predictions[input_base] = segment_predictions

    return predictions


def merge_segment_times(times, input_file, segment_times):
    '''
    Update the segment times into the time records.
    '''
    input_base = os.path.basename(input_file)
    times[input_base] = segment_times

    return times


def save_prediction_to_file(predictions, output_prediction_file):
    '''
    Save the prediction result to the json file.
    '''
    with open(output_prediction_file, "w") as f:
        f.write(json.dumps(predictions, indent=4))


def save_time_to_file(times, output_times_file):
    '''
    Save the evaluation time to the json file.
    '''
    with open(output_times_file, "w") as f:
        f.write(json.dumps(times, indent=4))


def save_frame_predictions(output_file, predictions):
    '''
    Save the predictions for a frame.
    bbox: [x_min, y_min, x_max, y_max, objectness, class]
    :return:
    '''
    write_flag = False
    with open(output_file, 'w') as f:
        for bbox in predictions:
            f.write(str(int(bbox[5])) + ' ' + str(bbox[4]) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(
                bbox[2]) + ' ' + str(bbox[3]) + '\n')
            write_flag = True

    if not write_flag:
        Path(output_file).touch()
