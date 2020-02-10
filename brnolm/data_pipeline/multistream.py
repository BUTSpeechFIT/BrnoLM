import torch


class LineTooLongError(Exception):
    pass


def batchify(data, bsz, cuda):
    """ For simple rearranging of 'single sentence' data.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


class Batcher:
    """Groups sequences in a list into batches
    """
    def __init__(self, samples, max_batch_size=None, max_total_len=None):
        self.max_batch_size = max_batch_size
        self.max_total_len = max_total_len
        self.samples = samples

    def _batch_size_ok(self, i, j):
        if not self.max_batch_size:
            return True

        return (j-i) <= self.max_batch_size

    def _total_len_ok(self, i, j):
        if not self.max_total_len:
            return True

        return max((len(t) for t in self.samples[i:j]), default=0) * (j-i) <= self.max_total_len

    def __iter__(self):
        i = 0
        while i < len(self.samples):
            j = i
            while j <= len(self.samples) and self._batch_size_ok(i, j) and self._total_len_ok(i, j):
                j += 1
            j -= 1

            if i == j:
                raise LineTooLongError(f'Failed to construct a batch on line {i} (zero-based)')

            batch = self.samples[i:j]
            assert len(batch) > 0
            assert sum(len(s) for s in batch) > 0

            yield batch
            i = j


class BatchBuilder():
    def __init__(self, streams, max_batch_size, discard_h=True):
        """ For complex combination of different lenghts sources.
        """
        self._streams = streams

        if max_batch_size <= 0:
            raise ValueError("BatchBuilder must be constructed"
                "with a positive batch size, (got {})".format(max_batch_size)
            )
        self._max_bsz = max_batch_size
        self._discard_h = discard_h

    def __iter__(self):
        streams = [iter(s) for s in self._streams]
        active_streams = []
        reserve_streams = streams

        while True:
            batch = []
            streams_continued = []
            streams_ended = []
            for i, s in enumerate(active_streams):
                try:
                    batch.append(next(s))
                    streams_continued.append(i)
                except StopIteration:
                    streams_ended.append(i)

            active_streams = [active_streams[i] for i in streams_continued]

            # refill the batch (of active streams)
            while len(reserve_streams) > 0:
                if len(batch) == self._max_bsz:
                    break

                stream = reserve_streams[0]
                del reserve_streams[0]
                try:
                    batch.append(next(stream))
                    active_streams.append(stream)
                except StopIteration:
                    pass

            if len(batch) == 0:
                return

            if self._discard_h:
                hs_passed_on = streams_continued
            else:
                hs_passed_on = (streams_continued + streams_ended)[:len(batch)]

            parts = zip(*batch)
            parts = [torch.stack(part) for part in parts]
            yield tuple(parts) + (torch.LongTensor(hs_passed_on), )
