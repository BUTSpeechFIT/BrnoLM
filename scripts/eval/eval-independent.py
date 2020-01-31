#!/usr/bin/env python

import argparse
import math
import torch

from brnolm.data_pipeline.reading import get_independent_lines
from brnolm.data_pipeline.threaded import OndemandDataProvider
from brnolm.data_pipeline.multistream import batcher
from brnolm.runtime.runtime_utils import init_seeds


if __name__ == '__main__':
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

    print("preparing data...")
    with open(args.data) as f:
        lines = get_independent_lines(f, lm.vocab)

    if args.sort_by_len:
        lines = sorted(lines, key=lambda l: len(l))

    nb_words = sum(len(ids) for ids in lines)
    nb_oovs = sum(sum(ids == lm.vocab.unk_ind).detach().item() for ids in lines)
    print('Nb oovs: {} / {} ({:.2f} %)\n'.format(nb_oovs, nb_words, 100.0 * nb_oovs/nb_words))

    loss = 0.0
    data_stream = OndemandDataProvider(batcher(lines, args.batch_size, args.max_tokens), cuda=False)
    total_actual_size = 0
    with torch.no_grad():
        for i, batch in enumerate(data_stream):
            act_batch_size = max(len(t) for t in batch) * len(batch)
            total_actual_size += act_batch_size
            per_line_losses = lm.batch_nll_idxs(batch, not args.prefix)
            loss += per_line_losses.sum().detach().item()

    print(f'Utilization: {100.0*nb_words/total_actual_size:.2f} % ({nb_words} words / {total_actual_size} softmaxes total)')
    print('total loss {:.1f} | loss {:5.2f} | ppl {:8.2f}'.format(loss, loss/nb_words, math.exp(loss/nb_words)))
