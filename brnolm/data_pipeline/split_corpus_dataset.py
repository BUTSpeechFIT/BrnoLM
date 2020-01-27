import torch

from brnolm.data_pipeline.temporal_splitting import TemporalSplits


class TokenizedSplitFFBase():
    def __init__(self, f, vocab, temporal_split_builder):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        sentence = f.read()
        self._words = sentence.split()
        self._tokens = torch.LongTensor([vocab[w] for w in self._words])

        self._temp_splits = temporal_split_builder(self._tokens)

    def __iter__(self):
        for x, t in self._temp_splits:
            yield x, t

    def __len__(self):
        return len(self._temp_splits)

    def input_words(self):
        for lend, rend in self._temp_splits.ranges():
            yield " ".join(self._words[lend:rend])


class TokenizedSplit(TokenizedSplitFFBase):
    def __init__(self, f, vocab, unroll_length):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        ts_builder = lambda seq: TemporalSplits(seq, nb_inputs_necessary=1, nb_targets_parallel=unroll_length)
        super().__init__(f, vocab, ts_builder)


class TokenizedSplitSingleTarget(TokenizedSplitFFBase):
    def __init__(self, f, vocab, unroll_length):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        ts_builder = lambda seq: TemporalSplits(seq, nb_inputs_necessary=unroll_length, nb_targets_parallel=1)
        super().__init__(f, vocab, ts_builder)


class TokenizedSplitFFMultiTarget(TokenizedSplitFFBase):
    def __init__(self, f, vocab, hist_len, nb_targets_parallel):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        ts_builder = lambda seq: TemporalSplits(seq, nb_inputs_necessary=hist_len, nb_targets_parallel=nb_targets_parallel)
        super().__init__(f, vocab, ts_builder)


class DomainAdaptationSplitFFBase:
    def __init__(self, f, vocab, end_portion, ts_builder):
        sentence = f.read()
        words = sentence.split()

        nb_domain_words = int(len(words)*end_portion-0.01)

        self._tokens = torch.LongTensor([vocab[w] for w in words[:-nb_domain_words]])
        self._domain_string = " ".join(words[len(words)-nb_domain_words:])

        self._temp_splitter = ts_builder(self._tokens)

    def __iter__(self):
        for x, t in self._temp_splitter:
            yield x, t

    def __len__(self):
        return len(self._temp_splitter)

    def input_words(self):
        return [self._domain_string]


class DomainAdaptationSplitFFMultiTarget(DomainAdaptationSplitFFBase):
    def __init__(self, f, vocab, hist_len, nb_targets_parallel, end_portion):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """
        ts_builder = lambda seq: TemporalSplits(seq, nb_inputs_necessary=hist_len, nb_targets_parallel=nb_targets_parallel)
        super().__init__(f, vocab, end_portion, ts_builder)


class DomainAdaptationSplit(DomainAdaptationSplitFFBase):
    def __init__(self, f, vocab, unroll_length, end_portion):
        """
            Args:
                f (file): File with a document.
                vocab (Vocabulary): Vocabulary for translation word -> index
        """

        ts_builder = lambda seq: TemporalSplits(seq, nb_inputs_necessary=unroll_length, nb_targets_parallel=1)
        super().__init__(f, vocab, end_portion, ts_builder)
