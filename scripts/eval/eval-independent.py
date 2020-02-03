#!/usr/bin/env python

import argparse
import logging
import math
import torch

from brnolm.data_pipeline.reading import get_independent_lines
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.data_pipeline.multistream import batcher
from brnolm.runtime.runtime_utils import init_seeds


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
        if prefix:
            logging.debug('Adding prefixes...')
            prefix_ind = self.lm.vocab[prefix]
            if prefix_ind == self.lm.vocab.unk_ind:
                logging.warning('Warning: prefix translates to unk!')

            prefix = torch.tensor([prefix_ind], dtype=self.lines[0].dtype)
            lines = [torch.cat([prefix, l]) for l in self.lines]

        loss = 0.0
        data_stream = OndemandDataProvider(batcher(lines, self.max_batch_size, self.max_tokens), cuda=False)
        total_actual_size = 0
        with torch.no_grad():
            for i, batch in enumerate(data_stream):
                per_line_losses = self.lm.batch_nll_idxs(batch, not prefix)
                loss += per_line_losses.sum().detach().item()
                total_actual_size += per_line_losses.numel()

        utilization = self.nb_tokens/total_actual_size
        return loss, utilization


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s::%(name)s] %(message)s')
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--prefix', type=str,
                        help='')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--max-tokens', type=int, default=1000, metavar='N',
                        help='Maximal number of softmaxes in a batch')
    parser.add_argument('--sort-by-len', action='store_true',
                        help='sort lines by len for better utilization')

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
    lm = torch.load(args.load, map_location='cpu')
    lm.nb_nonzero_masks = 0
    lm.eval()
    if args.cuda:
        lm.cuda()
    print(lm)

    evaluator = IndependentLinesEvaluator(lm, args.data, args.batch_size, args.max_tokens)
    loss, utilization = evaluator.evaluate(args.prefix)

    print(f'Utilization: {100.0*utilization:.2f} %')
    print('total loss {:.1f} | per token loss {:5.2f} | ppl {:8.2f}'.format(loss, loss/evaluator.nb_tokens, math.exp(loss/evaluator.nb_tokens)))


if __name__ == '__main__':
    main()
