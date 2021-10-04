import time
import numpy as np
from scheduling_utils.PriorityQueue import PriorityQueue

size_pool = dict()


class GeneralObjectBuffer:
    def __init__(self):
        '''
        This object buffer class is general, and can hold any image sizes. =
        Time profile: image size --> batch size --> stage
        '''
        self.buffer = PriorityQueue()

    def is_empty(self):
        '''
        Check whether the buffer is empty.
        :return:
        '''
        return self.buffer.is_empty()

    def push_new_object(self, new_object):
        '''
        When turn into a new period, push newly arrived objects into the buffer.
        '''
        self.buffer.push(new_object, priority=new_object.criticality)

    def execute_one_object_with_tubelet(self, current_time, model):
        '''
        Execute the selected object for just one stage.
        '''
        if self.buffer.is_empty():
            return current_time
        else:
            candidate_object = self.buffer.pop()

        # execute the selected tubelet with its newest object
        tubelet = candidate_object.tubelet

        # skip the finished tubelet or overdued tubelet
        if tubelet.is_finished or current_time > tubelet.deadline:
            if current_time > tubelet.deadline:
                tubelet.is_dead = True
            return current_time
        else:
            tubelet_agent_object = candidate_object.tubelet.newest_object

            # execute one object
            current_time, pred_box = tubelet_agent_object.object_inference(current_time, model)

            # execute the tubelet, only save the results if the due is not passed
            if tubelet_agent_object.is_finished:
                tubelet.is_finished = True
                tubelet.predictions = pred_box
                tubelet.finish_time = current_time
            else:
                tubelet.is_dead = True

            return current_time

    def execute_one_batch_with_tubelet(self, current_time, model):
        '''
        Execute the selected object for just one stage.
        '''
        if self.buffer.is_empty():
            return current_time
        else:
            candidate_object = self.buffer.pop()

        # execute the selected tubelet with its newest object
        tubelet = candidate_object.tubelet

        # skip the finished tubelet or overdued tubelet
        if tubelet.is_finished or current_time > tubelet.deadline:
            if current_time > tubelet.deadline:
                tubelet.is_dead = True
            return current_time
        else:
            tubelet_agent_object = tubelet.newest_object
            objects_to_infer = [tubelet_agent_object]

            # fetch object with same size
            while True:
                if self.buffer.is_empty():
                    break
                else:
                    next_object = self.buffer.pop()
                    next_tubelet = next_object.tubelet

                    if next_tubelet.is_finished or current_time > next_tubelet.deadline:
                        next_tubelet.is_dead = True
                    else:
                        next_agent_object = next_tubelet.newest_object

                        if next_agent_object.new_size == tubelet_agent_object.new_size:
                            objects_to_infer.append(next_agent_object)
                        else:
                            self.push_new_object(next_agent_object)
                            break

            # formulate the batch
            image_list = [obj.image for obj in objects_to_infer]
            image_list = np.concatenate(image_list, axis=0)
            print(np.shape(image_list))
            size = np.shape(image_list)
            if size not in size_pool:
                size_pool[size] = 1
                model.predict_on_batch(image_list)

            # execute one batch
            start = time.time()

            '''
            * input: [batch, width, height, 3 (rgb)]
            * output: list of 3 ndarray, related to 3 scales (downsampling rate: 8, 16, 32).
                      for each scale, [batch_size, x, y, 3, (4 + 1 + 80)], x, y are downsampled dimensions;
                      3 boxes at each scale for each pixel; 4 bounding box offsets, 1 objectness predictoin;
                      80 class predictions in COCO.
            '''
            pred_bbox_list = model.predict_on_batch(image_list)

            end = time.time()
            duration = end - start
            current_time += duration
            print(duration)

            # assign predictions to objects
            for i, object in enumerate(objects_to_infer):
                pred_bbox = [x[i] for x in pred_bbox_list]
                object.save_inference_result(pred_bbox, current_time)

            return current_time

    def print_buffer_state(self):
        '''
        Print the current state of this buffer.
        '''
        print("---------------------------------")
        print('Current buffer size: %s' % self.buffer.get_buffer_size())
