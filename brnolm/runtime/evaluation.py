import logging
import math
import numpy as np
from dataclasses import dataclass

from brnolm.data_pipeline.reading import get_independent_lines, tokens_from_fn
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.data_pipeline.multistream import Batcher, batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits

from brnolm.runtime.runtime_utils import TransposeWrapper
from brnolm.runtime.runtime_multifile import repackage_hidden

from brnolm.data_pipeline.aug_paper_pipeline import Corruptor, form_input_targets, LazyBatcher, TemplSplitterClean
from brnolm.runtime.runtime_utils import CudaStream

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
    def __init__(self, lm, fn_evalset, max_batch_size, max_tokens, logger=None, total_vocab_size=None):
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

        self.oov_cost_applicator = OovCostApplicator(
            get_oov_additional_cost(len(self.lm.vocab), total_vocab_size) if total_vocab_size else 0.0,
            lm.vocab.unk_ind,
        )

    def evaluate(self, prefix):
        self.lm.eval()
        h0_provider = self.lm.get_custom_h0_provider(prefix.split())

        loss = 0.0
        data_stream = OndemandDataProvider(Batcher(self.lines, self.max_batch_size, self.max_tokens), device=self.lm.device)
        total_actual_size = 0
        with torch.no_grad():
            for i, batch in enumerate(data_stream):
                per_line_losses = self.lm.batch_nll_idxs(batch, h0_provider)
                for i, (idxs, losses) in enumerate(zip(batch, per_line_losses)):
                    per_line_losses[i] = self.oov_cost_applicator(idxs, losses)
                loss += per_line_losses.sum().detach().item()
                total_actual_size += per_line_losses.numel()

        utilization = self.nb_tokens/total_actual_size
        return EvaluationReport(loss, self.nb_tokens, utilization)


class EnblockEvaluator:
    def __init__(self, lm, data_fn, batch_size, target_seq_len, logger=None, tokenize_regime='words'):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('EnblockEvaluator')
        self.batch_size = batch_size
        self.lm = lm

        ids = tokens_from_fn(data_fn, lm.vocab, regime=tokenize_regime, randomize=False)
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


class SubstitutionalEnblockEvaluator:
    def __init__(self, lm, data_fn, batch_size, target_seq_len, corruptor, nb_rounds, logger=None, tokenize_regime='words'):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('SubstitutionalEnblockEvaluator')
        self.batch_size = batch_size
        self.lm = lm
        self.nb_rounds = nb_rounds

        ids = tokens_from_fn(data_fn, lm.vocab, regime=tokenize_regime, randomize=False)
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
        self.data = corruptor(TransposeWrapper(data_tb))

    def evaluate(self, report_individual=False):
        overall_total_loss = 0.0
        overall_total_timesteps = 0.0
        ppls = []

        for round_no in range(self.nb_rounds):
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

            eval_report = EvaluationReport(total_loss.item(), total_timesteps, 1.0)
            ppl = math.exp(eval_report.loss_per_token)
            if report_individual:
                self.logger.info('total loss {:.1f} | per token loss {:5.2f} | ppl {:8.2f}'.format(eval_report.total_loss, eval_report.loss_per_token, ppl))

            overall_total_loss += total_loss
            overall_total_timesteps += total_timesteps
            ppls.append(ppl)

        ppls = np.asarray(ppls)
        self.logger.info(f'PPLs summary: {np.min(ppls):.2f} / {np.mean(ppls):.2f} / {np.max(ppls):.2f} , stddev: {np.std(ppls):.3f}')
        return EvaluationReport(overall_total_loss.item(), overall_total_timesteps, 1.0)


class SubstitutionalEnblockEvaluator_v2:
    def __init__(self, lm, data_fn, batch_size, target_seq_len, corruptor, nb_rounds, logger=None, tokenize_regime='words'):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('SubstitutionalEnblockEvaluator_v2')
        self.batch_size = batch_size
        self.lm = lm
        self.nb_rounds = nb_rounds

        ids = tokens_from_fn(data_fn, lm.vocab, regime=tokenize_regime, randomize=False)
        oov_mask = ids == lm.vocab.unk_ind
        nb_oovs = oov_mask.sum().item()

        nb_tokens = len(ids)
        oov_msg = 'Nb oovs: {} / {} ({:.2f} %)\n'.format(nb_oovs, len(ids), 100.0 * nb_oovs/nb_tokens)
        if nb_oovs / nb_tokens > 0.05:
            self.logger.warning(oov_msg)
        else:
            self.logger.info(oov_msg)

        streams = form_input_targets(ids)
        corrupted_provider = corruptor(streams)
        batch_former = LazyBatcher(batch_size, corrupted_provider)
        data_tb = TemplSplitterClean(target_seq_len, batch_former)

        self.data = CudaStream(TransposeWrapper(data_tb))

    def evaluate(self, report_individual=False):
        overall_total_loss = 0.0
        overall_total_timesteps = 0.0
        ppls = []

        for round_no in range(self.nb_rounds):
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

            eval_report = EvaluationReport(total_loss.item(), total_timesteps, 1.0)
            ppl = math.exp(eval_report.loss_per_token)
            if report_individual:
                self.logger.info('total loss {:.1f} | per token loss {:5.2f} | ppl {:8.2f}'.format(eval_report.total_loss, eval_report.loss_per_token, ppl))

            overall_total_loss += total_loss
            overall_total_timesteps += total_timesteps
            ppls.append(ppl)

        ppls = np.asarray(ppls)
        self.logger.info(f'PPLs summary: {np.min(ppls):.2f} / {np.mean(ppls):.2f} / {np.max(ppls):.2f} , stddev: {np.std(ppls):.3f}')
        return EvaluationReport(overall_total_loss.item(), overall_total_timesteps, 1.0)


def get_oov_additional_cost(lm_vocab_size, total_vocab_size):
    nb_oovs_uncovered = total_vocab_size - lm_vocab_size
    return math.log(nb_oovs_uncovered)


class OovCostApplicator:
    def __init__(self, oov_penalty, unk_ind):
        self.oov_penalty = oov_penalty
        self.unk_ind = unk_ind

    def __call__(self, ids, losses):
        if self.oov_penalty == 0.0:
            return losses

        unk_mask = (ids == self.unk_ind).to(losses.device)
        zero_padding = torch.zeros(
            (len(losses) - len(ids),),
            dtype=unk_mask.dtype, device=unk_mask.device
        )
        unk_mask = torch.cat([unk_mask, zero_padding])

        return losses + unk_mask*self.oov_penalty
