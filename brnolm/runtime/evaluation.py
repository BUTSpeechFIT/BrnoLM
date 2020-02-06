import logging
from brnolm.data_pipeline.reading import get_independent_lines
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.data_pipeline.multistream import batcher

import torch


class IndependentLinesEvaluator:
    def __init__(self, lm, fn_evalset, max_batch_size, max_tokens, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('IndependentLinesEvaluator')
        self.lm = lm
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens

        self.logger.debug("preparing data...")
        with open(fn_evalset) as f:
            self.lines = get_independent_lines(f, lm.vocab)

        self.logger.debug("sorting lines...")
        self.lines = sorted(self.lines, key=lambda l: len(l))

        self.logger.debug("computing statistics...")
        self.nb_tokens = sum(len(ids) for ids in self.lines)
        nb_oovs = sum(sum(ids == lm.vocab.unk_ind).detach().item() for ids in self.lines)
        oov_msg = 'Nb oovs: {} / {} ({:.2f} %)\n'.format(nb_oovs, self.nb_tokens, 100.0 * nb_oovs/self.nb_tokens)
        if nb_oovs / self.nb_tokens > 0.05:
            self.logger.warning(oov_msg)
        else:
            self.logger.info(oov_msg)

    def evaluate(self, prefix):
        h0_provider = self.lm.get_custom_h0_provider(prefix.split())

        loss = 0.0
        data_stream = OndemandDataProvider(batcher(self.lines, self.max_batch_size, self.max_tokens), cuda=False)
        total_actual_size = 0
        with torch.no_grad():
            for i, batch in enumerate(data_stream):
                per_line_losses = self.lm.batch_nll_idxs(batch, h0_provider)
                loss += per_line_losses.sum().detach().item()
                total_actual_size += per_line_losses.numel()

        utilization = self.nb_tokens/total_actual_size
        return loss, utilization
