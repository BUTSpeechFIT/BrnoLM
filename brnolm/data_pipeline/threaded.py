import threading
import queue


class DataCreator(threading.Thread):
    def __init__(self, q, data_stream, cuda):
        super().__init__(daemon=True)
        self.q = q
        self.in_stream = data_stream
        self.cuda = cuda

    def run(self):
        for batch in self.in_stream:
            if self.cuda:
                batch = (x.cuda() for x in batch)

            self.q.put(batch)


class OndemandDataProvider:
    def __init__(self, in_data, cuda):
        self.data = in_data
        self.cuda = cuda

    def __iter__(self):
        q = queue.Queue(maxsize=10)
        feeder_thread = DataCreator(q, self.data, self.cuda)
        feeder_thread.start()

        while feeder_thread.is_alive() or not q.empty():
            yield q.get()
