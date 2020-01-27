class TemporalSplits():
    def __init__(self, seq, nb_inputs_necessary, nb_targets_parallel):
        self._seq = seq
        self._nb_inputs_necessary = nb_inputs_necessary
        self._nb_target_parallel = nb_targets_parallel

    def __iter__(self):
        for lend, rend in self.ranges():
            yield (
                self._seq[lend:rend],
                self._seq[lend+self._nb_inputs_necessary:rend+1]
            )

    def __len__(self):
        return max(len(self._seq) - self._nb_inputs_necessary - self._nb_target_parallel + 1, 0)

    def ranges(self):
        for i in range(0, len(self), self._nb_target_parallel):
            lend = i
            rend = i + self._nb_inputs_necessary + self._nb_target_parallel - 1
            yield lend, rend
