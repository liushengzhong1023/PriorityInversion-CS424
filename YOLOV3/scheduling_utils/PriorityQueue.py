import heapq


class PriorityQueue:
    '''
    Source: https://howtodoinjava.com/python/priority-queue/
    NOTE: Higher priority itmes pop first.
    '''

    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        '''
        Push item into the queue.
        '''
        if type(priority) is tuple:
            saved_priority = tuple([-item for item in priority])
            heapq.heappush(self._queue, (saved_priority, self._index, item))
        else:
            heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        '''
        Pop intem from the queue.
        '''
        return heapq.heappop(self._queue)[-1]

    def is_empty(self):
        '''
        Judge whether the queue is empty.
        '''
        return len(self._queue) == 0

    def pop_all_item_list(self):
        '''
        Return the queue as a list.
        '''
        output_list = []

        while not self.is_empty():
            output_list.append(self.pop())

        return output_list

    def get_buffer_size(self):
        '''
        Return the buffer size
        '''
        return len(self._queue)