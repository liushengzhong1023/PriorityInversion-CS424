class ObjectTubelet:
    def __init__(self, tubelet_id, tube_object):
        '''
        Create the tubelet with an id and the first tube_object.
        The first object of the tubelet will be used to join the queue, while the newest_object will be executed.
        '''
        self.tubelet_id = tubelet_id
        self.object_list = [tube_object]
        self.newest_object = tube_object

        # store the predicted class
        self.predictions = None

        # store whether the tubelet is finished
        self.is_finished = False
        self.finish_time = None

        # A tubelet is dead if its deadline has passed when its newest object is executed.
        # The dead tubelet will not be able to take new objects.
        self.is_dead = False

        # tubelet deadline
        self.deadline = tube_object.deadline

    def add_new_object(self, new_object):
        '''
        Add a new object to the tubelet. The newest object is fixed as long as any object has been executed.
        '''
        # update the deadline every time the new frame comes
        if new_object.deadline > self.deadline and not self.is_finished:
            self.deadline = new_object.deadline
            self.newest_object = new_object

        self.object_list.append(new_object)

    def set_finish_time(self, finish_time, predictions, agent_obj):
        '''
        Set finish time of all objects within the tubelet.
        :param finish_time:
        :return:
        '''
        # update tubelet status
        self.is_finished = True
        self.predictions = predictions
        self.finish_time = finish_time

        # update each object for this tubelet, copy the predictions.
        for obj in self.object_list:
            if not obj.is_finished:
                obj.is_finished = True
                obj.finish_time = finish_time

                # size mapping on the predictions, bbox: [x_min, y_min, x_max, y_max, objectness, class]
                obj.copy_predictions(agent_obj)


    def set_death(self):
        '''
        Set death of all objects within the tubelet.
        :param finish_time:
        :return:
        '''
        self.is_dead = True

        for obj in self.object_list:
            obj.is_overdue = True
