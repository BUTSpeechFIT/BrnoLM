#!/usr/bin/env python

import argparse
import logging
import math
import torch

from brnolm.runtime.safe_gpu import GPUOwner
from brnolm.runtime.runtime_utils import init_seeds
from brnolm.runtime.evaluation import SubstitutionalEnblockEvaluator_v2
from brnolm.data_pipeline.aug_paper_pipeline import Corruptor


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s::%(name)s] %(message)s')
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--shuffle-lines', action='store_true',
                        help='shuffle lines before every epoch')

    parser.add_argument('--subs-rate', type=float, required=True,
                        help='what ratio of input tokens should be substituted')
    parser.add_argument('--del-rate', type=float, required=True,
                        help='what ratio of tokens should be deleted')
    parser.add_argument('--ins-rate', type=float, required=True,
                        help='what ratio of tokens should be inserted')
    parser.add_argument('--rounds', type=int, required=True,
                        help='how many times to run through the eval data')
    parser.add_argument('--individual', action='store_true',
                        help='report individual rounds')

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
    if args.cuda:
        gpu_owner = GPUOwner(lambda: torch.zeros((1), device='cuda'))

    lm = torch.load(args.load, map_location=device)
    print(lm)

    evaluator = SubstitutionalEnblockEvaluator_v2(
        lm,
        args.data,
        args.batch_size,
        args.target_seq_len,
        lambda streams: Corruptor(streams, args.subs_rate, len(lm.vocab), args.del_rate, args.ins_rate, protected=[lm.vocab['</s>']]),
        args.rounds,
    )
    eval_report = evaluator.evaluate(report_individual=args.individual)

    print('total loss {:.1f} | per token loss {:5.2f} | ppl {:8.2f}'.format(eval_report.total_loss, eval_report.loss_per_token, math.exp(eval_report.loss_per_token)))
