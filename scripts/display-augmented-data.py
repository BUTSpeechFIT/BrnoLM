#!/usr/bin/env python3

import argparse
import logging
import pickle
import sys
import torch

from brnolm.data_pipeline.reading import tokens_from_fn
from brnolm.data_pipeline.aug_paper_pipeline import form_input_targets
from brnolm.data_pipeline.aug_paper_pipeline import Corruptor
from brnolm.data_pipeline.aug_paper_pipeline import Confuser, StatisticsCorruptor


RED_MARK = '\033[91m'
END_MARK = '\033[0m'


def main(args):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s::%(name)s] %(message)s')

    lm = torch.load(args.load, map_location='cpu')

    tokenize_regime = 'words'
    train_ids = tokens_from_fn(args.data, lm.vocab, randomize=False, regime=tokenize_regime)
    train_streams = form_input_targets(train_ids)
    if args.statistics:
        with open(args.statistics, 'rb') as f:
            summary = pickle.load(f)
        confuser = Confuser(summary.confusions, lm.vocab, mincount=5)
        corrupted_provider = StatisticsCorruptor(train_streams, confuser, args.ins_rate, protected=[lm.vocab['</s>']])
    else:
        corrupted_provider = Corruptor(train_streams, args.subs_rate, len(lm.vocab), args.del_rate, args.ins_rate, protected=[lm.vocab['</s>']])

    inputs, targets = corrupted_provider.provide()

    for i in range(args.nb_tokens):
        in_word = lm.vocab.i2w(inputs[i].item())
        target_word = lm.vocab.i2w(targets[i].item())

        is_error = i > 0 and inputs[i] != targets[i-1]
        if args.color and is_error:
            sys.stdout.write(f'{RED_MARK}{in_word}{END_MARK} {target_word}\n')
        else:
            sys.stdout.write(f'{in_word} {target_word}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='location of the train corpus')
    parser.add_argument('--nb-tokens', type=int, default=10,
                        help='how many input-target pairs to show')
    parser.add_argument('--color', action='store_true',
                        help='Use ANSI colorcodes to highlight errors')

    parser.add_argument('--ins-rate', type=float, required=True,
                        help='what ratio of tokens should be inserted')
    parser.add_argument('--subs-rate', type=float,
                        help='what ratio of input tokens should be randomly')
    parser.add_argument('--del-rate', type=float,
                        help='what ratio of tokens should be removed')
    parser.add_argument('--statistics', type=str,
                        help='Use these statistics to determine exact mistakes')

    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()
    print(args)

    if (args.del_rate is None or args.subs_rate is None) and not args.statistics:
        sys.stderr.write('either (--del-rate and --subs-rate) or (--statistics) must be provided\n')
        sys.exit(2)

    if (args.del_rate is not None or args.subs_rate is not None) and args.statistics:
        sys.stderr.write('(--del-rate and --subs-rate) and (--statistics) are mutually exclusive\n')
        sys.exit(2)

    main(args)
