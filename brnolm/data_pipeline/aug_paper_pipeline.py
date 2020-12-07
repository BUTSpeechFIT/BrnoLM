import numpy as np
import random
import torch


def form_input_targets(stream):
    return stream[:-1], stream[1:]


class CleanStreamsProvider:
    def __init__(self, stream):
        self.stream = stream

    def provide(self):
        return self.stream[:-1], self.stream[1:]


class Corruptor:
    def __init__(self, streams_provider, subs_rate, subs_range, del_rate, ins_rate, protected=[]):
        self.streams_provider = streams_provider
        self.sr = subs_rate
        self.subs_range = subs_range
        self.dr = del_rate
        self.ir = ins_rate
        self.protected = protected

    def provide(self):
        streams = self.streams_provider.provide()
        assert len(streams[0]) == len(streams[1])
        self.inputs = streams[0]
        self.targets = streams[1]

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


class InputTargetCorruptor:
    def __init__(self, streams_provider, input_subs_rate, target_subs_rate, subs_range, del_rate, ins_rate, protected=[]):
        self.streams_provider = streams_provider
        self.isr = input_subs_rate
        self.tsr = target_subs_rate
        self.subs_range = subs_range
        self.dr = del_rate
        self.ir = ins_rate
        self.protected = protected

    def provide(self):
        streams = self.streams_provider.provide()
        assert len(streams[0]) == len(streams[1])
        self.inputs = streams[0]
        self.targets = streams[1]

        inputs = []
        targets = []

        i = 0
        nb_nonprotected = 0
        nb_subs = 0
        nb_target_subs = 0
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
            elif roll < self.dr + self.isr:
                targets.append(self.targets[i])
                inputs.append(random.randrange(self.subs_range))
                i += 1
                nb_subs += 1
            elif roll < self.dr + self.isr:
                targets.append(self.targets[i])
                inputs.append(random.randrange(self.subs_range))
                i += 1
                nb_subs += 1
            elif roll < self.dr + self.isr + self.ir:
                targets.append(self.targets[i])
                inputs.append(random.randrange(self.subs_range))
                nb_inss += 1
            elif roll < self.dr + self.isr + self.ir + self.tsr:
                targets.append(random.randrange(self.subs_range))
                inputs.append(self.inputs[i])
                i += 1
                nb_target_subs += 1
            else:
                targets.append(self.targets[i])
                inputs.append(self.inputs[i])
                i += 1

        print(f'len {len(self.inputs)}, proper {nb_nonprotected}| D: {100.0*nb_dels/nb_nonprotected:.2f} % ({nb_dels}) S: {100.0*nb_subs/nb_nonprotected:.2f} % S_t: {100.0*nb_target_subs/nb_nonprotected:.2f} % ({nb_subs}) I: {100.0*nb_inss/nb_nonprotected:.2f} % ({nb_inss})')
        return torch.tensor(inputs), torch.tensor(targets)


class TargetCorruptor:
    def __init__(self, streams_provider, subs_rate, subs_range, del_rate, ins_rate, protected=[]):
        self.streams_provider = streams_provider
        self.sr = subs_rate
        self.subs_range = subs_range
        self.dr = del_rate
        self.ir = ins_rate
        self.protected = protected

    def provide(self):
        streams = self.streams_provider.provide()
        assert len(streams[0]) == len(streams[1])
        self.inputs = streams[0]
        self.targets = streams[1]

        inputs = []
        targets = []

        i = 0
        nb_nonprotected = 0
        nb_subs = 0
        nb_dels = 0
        nb_inss = 0
        while i < len(self.inputs):
            if self.targets[i] in self.protected:
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
                targets.append(random.randrange(self.subs_range))
                inputs.append(self.inputs[i])
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


def cut_counts(confusions, mincount):
    counts = {ref_word: {k: v for k, v in nums.items() if v >= mincount} for ref_word, nums in confusions.items()}
    total_counts = {ref_word: sum(counts[ref_word].values()) for ref_word in counts}

    return {ref_word: counts for ref_word, counts in confusions.items() if total_counts[ref_word] > 0}


class Sampler:
    def __init__(self, replacements, probs):
        self.replacements = replacements
        self.probs = probs

    def __call__(self, id, size):
        return np.random.choice(self.replacements[id], size, p=self.probs[id])


class SampleCache:
    def __init__(self, ids, replacements, probs, size=1000):
        self.replacements = replacements
        self.size = size
        self.probs = probs
        self.sampler = Sampler(replacements, probs)
        self.cache = {id: self.sampler(id, size) for id in ids}
        self.backup_cache = {id: self.sampler(id, size) for id in ids}
        self.next = {id: 0 for id in ids}

    def get_next(self, id):
        if self.next[id] == len(self.cache[id]):
            self.cache[id] = self.backup_cache[id]
            self.backup_cache[id] = self.sampler(id, self.size)
            self.next[id] = 0

        sample = self.cache[id][self.next[id]]
        self.next[id] += 1
        return sample


class Confuser:
    NONE_ID = -1

    def __init__(self, confusion_counts, vocab, mincount):
        probs = cut_counts(confusion_counts, mincount)

        def translate(word):
            if word is None:
                return Confuser.NONE_ID
            else:
                return vocab[word]

        id_counts = {translate(ref_word): {translate(replacement): v for replacement, v in stats.items()} for ref_word, stats in probs.items()}
        id_counts_np = {ref_id: (np.asarray(list(probs.keys())), np.asarray(list(probs.values()))) for ref_id, probs in id_counts.items()}

        self.samples_cache = SampleCache(
            list(id_counts_np.keys()),
            {id: repls for id, (repls, _) in id_counts_np.items()},
            {id: counts/np.sum(counts) for id, (_, counts) in id_counts_np.items()},
            size=1000,
        )

    def replace(self, token_id):
        if token_id is None:
            token_id = Confuser.NONE_ID

        try:
            sample = self.samples_cache.get_next(token_id)
        except KeyError:  # New token, never seen in stats. Typical for <unk> and other unks
            return token_id

        if sample == Confuser.NONE_ID:
            return None
        else:
            return sample


class StatisticsCorruptor:
    def __init__(self, streams_provider, confuser, ins_rate, protected=[]):
        self.streams_provider = streams_provider
        self.confuser = confuser
        self.ir = ins_rate
        self.protected = protected

    def provide(self):
        streams = self.streams_provider.provide()
        assert len(streams[0]) == len(streams[1])
        self.inputs = streams[0]
        self.targets = streams[1]

        inputs = []
        targets = []

        i = 0
        nb_nonprotected = 0
        nb_subs = 0
        nb_dels = 0
        nb_inss = 0

        in_len = len(self.inputs)
        while i < in_len:
            in_token = self.inputs[i]
            target_token = self.targets[i]
            if in_token in self.protected:
                targets.append(target_token)
                inputs.append(in_token)
                i += 1
                continue

            nb_nonprotected += 1

            roll = random.random()
            if roll < self.ir:  # Insertion
                insertion = self.confuser.replace(None)
                targets.append(target_token)
                inputs.append(insertion)
                nb_inss += 1
            else:
                repl = self.confuser.replace(in_token)
                if repl == in_token:  # Correct token
                    targets.append(target_token)
                    inputs.append(in_token)
                    i += 1
                elif repl is None:  # Deletion
                    i += 1
                    nb_dels += 1
                else:  # Substitution
                    targets.append(target_token)
                    inputs.append(repl)
                    i += 1
                    nb_subs += 1

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
