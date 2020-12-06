import threading
import queue


class DataCreator(threading.Thread):
    def __init__(self, q, data_stream, device):
        super().__init__(daemon=True)
        self.q = q
        self.in_stream = data_stream
        self.device = device

    def run(self):
        for batch in self.in_stream:
            batch = (x.to(self.device) for x in batch)

            self.q.put(batch)


class OndemandDataProvider:
    def __init__(self, in_data, device):
        self.data = in_data
        self.device = device

    def __iter__(self):
        q = queue.Queue(maxsize=10)
        feeder_thread = DataCreator(q, self.data, self.device)
        feeder_thread.start()

        while feeder_thread.is_alive() or not q.empty():
            yield q.get()
