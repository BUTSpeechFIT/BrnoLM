import torch
import random


def form_input_targets(stream):
    return stream[:-1], stream[1:]


class Corruptor:
    def __init__(self, streams, subs_rate, subs_range, del_rate, ins_rate, protected=[]):
        assert len(streams[0]) == len(streams[1])
        self.inputs = streams[0]
        self.targets = streams[1]
        self.sr = subs_rate
        self.subs_range = subs_range
        self.dr = del_rate
        self.ir = ins_rate
        self.protected = protected

    def provide(self):
        inputs = []
        targets = []

        i = 0
        nb_nonprotected = 0
        nb_subs = 0
        nb_dels = 0
        nb_inss = 0
        while i < len(self.inputs):
            if self.inputs[i] in self.protected:
                targets.append(self.targets[i])
                inputs.append(self.inputs[i])
                i += 1
                continue

            nb_nonprotected += 1

            roll = random.random()
            if roll < self.dr:
                i += 1
                nb_dels += 1
            elif roll < self.dr + self.sr:
                targets.append(self.targets[i])
                inputs.append(random.randrange(self.subs_range))
                i += 1
                nb_subs += 1
            elif roll < self.dr + self.sr + self.ir:
                targets.append(self.targets[i])
                inputs.append(random.randrange(self.subs_range))
                nb_inss += 1
            else:
                targets.append(self.targets[i])
                inputs.append(self.inputs[i])
                i += 1

        print(f'len {len(self.inputs)}, proper {nb_nonprotected}| D: {100.0*nb_dels/nb_nonprotected:.2f} % ({nb_dels}) S: {100.0*nb_subs/nb_nonprotected:.2f} % ({nb_subs}) I: {100.0*nb_inss/nb_nonprotected:.2f} % ({nb_inss})')
        return torch.tensor(inputs), torch.tensor(targets)


class LazyBatcher:
    def __init__(self, bsz, source):
        self.source = source
        self.bsz = bsz

    def provide(self):
        inputs_stream, targets_stream = self.source.provide()
        len(inputs_stream) == len(targets_stream)

        nb_batches = len(targets_stream) // self.bsz

        inputs_stream = inputs_stream.narrow(0, 0, nb_batches * self.bsz)
        targets_stream = targets_stream.narrow(0, 0, nb_batches * self.bsz)

        self.input_batches = inputs_stream.view(self.bsz, -1).t().contiguous()
        self.target_batches = targets_stream.view(self.bsz, -1).t().contiguous()
        return self.input_batches, self.target_batches


class TemplSplitterClean:
    def __init__(self, target_seq_len, batched_data_provider):
        self.data_provider = batched_data_provider
        self.tsl = target_seq_len

    def __iter__(self):
        data = self.data_provider.provide()
        i = 0
        while i < len(data[0]):
            yield (
                data[0][i:i+self.tsl],
                data[1][i:i+self.tsl],
            )
            i += self.tsl
