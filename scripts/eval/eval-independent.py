#!/usr/bin/env python

import argparse
import math
import torch

from brnolm.data_pipeline.reading import get_independent_lines
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.runtime.runtime_utils import init_seeds


def batcher(samples, batch_size):
    i = 0
    while i + batch_size - 1 < len(samples):
        yield samples[i:i+batch_size]
        i += batch_size

    if i < len(samples):
        yield samples[i:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--prefix', type=str,
                        help='')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')

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
    if args.cuda:
        lm.cuda()
    print(lm)

    print("preparing data...")
    with open(args.data) as f:
        lines = get_independent_lines(f, lm.vocab)

    nb_words = sum(len(ids) for ids in lines)
    nb_oovs = sum(sum(ids == lm.vocab.unk_ind).detach().item() for ids in lines)
    print('Nb oovs: {} / {} ({:.2f} %)\n'.format(nb_oovs, nb_words, 100.0 * nb_oovs/nb_words))

    loss = 0.0
    data_stream = OndemandDataProvider(batcher(lines, args.batch_size), cuda=False)
    for i, batch in enumerate(data_stream):
        per_line_losses = lm.batch_nll_idxs(batch, not args.prefix)
        loss += per_line_losses.sum().detach().item()

    print('total loss {:.1f} | loss {:5.2f} | ppl {:8.2f}'.format(loss, loss/nb_words, math.exp(loss/nb_words)))
