import logging
from dataclasses import dataclass

from brnolm.data_pipeline.reading import get_independent_lines, tokens_from_fn
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.data_pipeline.multistream import Batcher, batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits

from brnolm.runtime.runtime_utils import TransposeWrapper
from brnolm.runtime.runtime_multifile import repackage_hidden

import torch


@dataclass
class EvaluationReport:
    total_loss: float
    nb_words: int
    utilization: float

    @property
    def loss_per_token(self):
        return self.total_loss / self.nb_words


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
        self.lm.eval()
        h0_provider = self.lm.get_custom_h0_provider(prefix.split())

        loss = 0.0
        data_stream = OndemandDataProvider(Batcher(self.lines, self.max_batch_size, self.max_tokens), cuda=False)
        total_actual_size = 0
        with torch.no_grad():
            for i, batch in enumerate(data_stream):
                per_line_losses = self.lm.batch_nll_idxs(batch, h0_provider)
                loss += per_line_losses.sum().detach().item()
                total_actual_size += per_line_losses.numel()

        utilization = self.nb_tokens/total_actual_size
        return EvaluationReport(loss, self.nb_tokens, utilization)


class EnblockEvaluator:
    def __init__(self, lm, data_fn, batch_size, target_seq_len, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('IndependentLinesEvaluator')
        self.batch_size = batch_size
        self.lm = lm

        ids = tokens_from_fn(data_fn, lm.vocab, randomize=False)
        oov_mask = ids == lm.vocab.unk_ind
        nb_oovs = oov_mask.sum().item()

        nb_tokens = len(ids)
        oov_msg = 'Nb oovs: {} / {} ({:.2f} %)\n'.format(nb_oovs, len(ids), 100.0 * nb_oovs/nb_tokens)
        if nb_oovs / nb_tokens > 0.05:
            self.logger.warning(oov_msg)
        else:
            self.logger.info(oov_msg)

        batched = batchify(ids, batch_size, lm.device == torch.device('cuda:0'))
        data_tb = TemporalSplits(
            batched,
            nb_inputs_necessary=lm.model.in_len,
            nb_targets_parallel=target_seq_len
        )
        self.data = TransposeWrapper(data_tb)

    def evaluate(self):
        self.lm.eval()

        total_loss = 0.0
        total_timesteps = 0
        hidden = self.lm.model.init_hidden(self.batch_size)

        for X, targets in self.data:
            hidden = repackage_hidden(hidden)

            output, hidden = self.lm.model(X, hidden)
            losses = self.lm.decoder.neg_log_prob_raw(output, targets)

            total_loss += losses.sum().detach()
            total_timesteps += targets.numel()

        return EvaluationReport(total_loss.item(), total_timesteps, 1.0)
