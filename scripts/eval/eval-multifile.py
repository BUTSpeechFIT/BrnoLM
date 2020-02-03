#!/usr/bin/env python

import argparse
import math
import torch

from brnolm.data_pipeline.multistream import BatchBuilder

from brnolm.data_pipeline.reading import tokens_from_file
from brnolm.data_pipeline.temporal_splitting import TemporalSplits
from brnolm.runtime.runtime_utils import CudaStream, init_seeds, filelist_to_objects
from brnolm.runtime.runtime_multifile import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--file-list', type=str, required=True,
                        help='file with paths to training documents')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--target-seq-len', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--concat-articles', action='store_true',
                        help='pass hidden states over article boundaries')
    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading model...")
    lm = torch.load(args.load)
    if args.cuda:
        lm.cuda()
    print(lm.model)

    print("preparing data...")

    def temp_splits_from_fn(fn):
        tokens = tokens_from_file(fn, lm.vocab, randomize=False)
        return TemporalSplits(tokens, lm.model.in_len, args.target_seq_len)

    tss = filelist_to_objects(args.file_list, temp_splits_from_fn)
    data = BatchBuilder(tss, args.batch_size,
                        discard_h=not args.concat_articles)
    if args.cuda:
        data = CudaStream(data)

    loss = evaluate(lm, data, use_ivecs=False)
    print('loss {:5.2f} | ppl {:8.2f}'.format(loss, math.exp(loss)))
