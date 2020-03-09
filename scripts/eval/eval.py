#!/usr/bin/env python

import argparse
import math
import torch

from brnolm.data_pipeline.reading import tokens_from_fn
from brnolm.data_pipeline.multistream import batchify
from brnolm.data_pipeline.temporal_splitting import TemporalSplits

from brnolm.runtime.runtime_utils import TransposeWrapper, init_seeds
from brnolm.runtime.runtime_multifile import repackage_hidden
from brnolm.runtime.evaluation import EvaluationReport


class EnblockEvaluator:
    def __init__(self, lm, data):
        self.data = data

    def evaluate(self):
        lm.eval()

        total_loss = 0.0
        total_timesteps = 0
        hidden = lm.model.init_hidden(args.batch_size)

        for X, targets in self.data:
            hidden = repackage_hidden(hidden)

            output, hidden = lm.model(X, hidden)
            loss, nb_words = lm.decoder.neg_log_prob(output, targets)

            total_loss += loss.data
            total_timesteps += nb_words

        return EvaluationReport(total_loss.item(), total_timesteps, 1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')
    parser.add_argument('--characters', action='store_true',
                        help='work on character level, whitespace is significant')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35,
                        help='sequence length')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    lm = torch.load(args.load, map_location=device)
    print(lm)

    print("preparing data...")
    tokenize_regime = 'words'
    if args.characters:
        tokenize_regime = 'chars'

    ids = tokens_from_fn(args.data, lm.vocab, randomize=False, regime=tokenize_regime)
    batched = batchify(ids, 10, args.cuda)
    data_tb = TemporalSplits(
        batched,
        nb_inputs_necessary=lm.model.in_len,
        nb_targets_parallel=args.target_seq_len
    )
    data = TransposeWrapper(data_tb)

    oov_mask = ids == lm.vocab.unk_ind
    nb_oovs = oov_mask.sum()
    print('Nb oovs: {} ({:.2f} %)\n'.format(nb_oovs, 100.0 * nb_oovs/len(ids)))

    evaluator = EnblockEvaluator(lm, data)
    eval_report = evaluator.evaluate()

    print('total loss {:.1f} | per token loss {:5.2f} | ppl {:8.2f}'.format(eval_report.total_loss, eval_report.loss_per_token, math.exp(eval_report.loss_per_token)))
