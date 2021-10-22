import random
import torch


class SequenceReadingHead:
    def __init__(self, seq, start=0):
        self.seq = seq
        self.pos = start

    def __next__(self):
        val = self.seq[self.pos]
        self.pos = (self.pos + 1) % len(self.seq)
        return val


class FileReadingHead:
    def __init__(self, fn, pos, tokenizer, buffer_size=512):
        self.file = open(fn, 'rb')
        self.file.seek(pos)
        self.file.readline()  # move to next full line
        self.tokenizer = tokenizer

        self.buffer = []
        self.idx_in_buffer = 0
        self.target_buffer_size = buffer_size

    def __next__(self):
        assert self.idx_in_buffer <= len(self.buffer)

        if self.idx_in_buffer == len(self.buffer):
            self.refill_buffer()

        tok = self.buffer[self.idx_in_buffer]
        self.idx_in_buffer += 1

        return tok

    def refill_buffer(self):
        self.buffer.clear()
        while len(self.buffer) < self.target_buffer_size:
            line = self.file.readline().decode()
            if line == '':
                self.file.seek(0)
            self.buffer.extend(self.tokenizer(line))

        self.idx_in_buffer = 0


class StreamingCorruptor:
    def __init__(self, stream_provider, subs_rate, subs_range, del_rate, ins_rate, protected=[]):
        self.stream_provider = stream_provider
        self.sr = subs_rate
        self.subs_range = subs_range
        self.dr = del_rate
        self.ir = ins_rate
        self.protected = protected

        self.to_be_input = next(self.stream_provider)
        self.to_be_target = next(self.stream_provider)

    def __next__(self):
        nb_nonprotected = 0
        nb_subs = 0
        nb_dels = 0
        nb_inss = 0

        if self.to_be_input in self.protected:
            x, t = self.to_be_input, self.to_be_target
            self._move_one_token()
            return x, t

        nb_nonprotected += 1

        roll = random.random()
        if roll < self.dr:
            self._move_one_token()
            nb_dels += 1
            return next(self)
        elif roll < self.dr + self.sr:
            x, t = random.randrange(self.subs_range), self.to_be_target
            self._move_one_token()
            nb_subs += 1
            return x, t
        elif roll < self.dr + self.sr + self.ir:
            x, t = random.randrange(self.subs_range), self.to_be_target
            nb_inss += 1
            return x, t
        else:
            x, t = self.to_be_input, self.to_be_target
            self._move_one_token()
            return x, t

    def _move_one_token(self):
        self.to_be_input = self.to_be_target
        self.to_be_target = next(self.stream_provider)

    def summary(self):
        print(f'len {len(self.inputs)}, proper {nb_nonprotected}| D: {100.0*nb_dels/nb_nonprotected:.2f} % ({nb_dels}) S: {100.0*nb_subs/nb_nonprotected:.2f} % ({nb_subs}) I: {100.0*nb_inss/nb_nonprotected:.2f} % ({nb_inss})')


class BatchingSlicingIterator:
    def __init__(self, sources, seq_len):
        self.sources = sources
        self.seq_len = seq_len

    def __next__(self):
        samples = []
        for _ in range(self.seq_len):
            samples.append(list(next(s) for s in self.sources))

        inputs = torch.tensor([[s[0] for s in time_slice] for time_slice in samples])
        targets = torch.tensor([[s[1] for s in time_slice] for time_slice in samples])

        return inputs.permute(1, 0), targets.permute(1, 0).contiguous()
