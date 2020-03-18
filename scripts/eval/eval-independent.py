#!/usr/bin/env python

import argparse
import logging
import math
import torch

from brnolm.runtime.runtime_utils import init_seeds
from brnolm.runtime.evaluation import IndependentLinesEvaluator


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s::%(name)s] %(message)s')
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--prefix', type=str,
                        help='')
    parser.add_argument('--total-vocab-size', type=int,
                        help='how many words should be assumed to exist overall')

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--max-tokens', type=int, default=1000, metavar='N',
                        help='Maximal number of softmaxes in a batch')

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

    evaluator = IndependentLinesEvaluator(
        lm=lm,
        fn_evalset=args.data,
        max_batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        total_vocab_size=args.total_vocab_size
    )
    eval_report = evaluator.evaluate(args.prefix)

    print(f'Utilization: {100.0*eval_report.utilization:.2f} %')
    print('total loss {:.1f} | per token loss {:5.2f} | ppl {:8.2f}'.format(eval_report.total_loss, eval_report.loss_per_token, math.exp(eval_report.loss_per_token)))


if __name__ == '__main__':
    main()
