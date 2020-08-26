#!/usr/bin/env python3

import argparse
import logging
import torch

from brnolm.data_pipeline.reading import tokens_from_fn
from brnolm.data_pipeline.aug_paper_pipeline import Corruptor, form_input_targets


def main(args):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s::%(name)s] %(message)s')

    lm = torch.load(args.load, map_location='cpu')

    tokenize_regime = 'words'
    train_ids = tokens_from_fn(args.data, lm.vocab, randomize=False, regime=tokenize_regime)
    train_streams = form_input_targets(train_ids)
    corrupted_provider = Corruptor(train_streams, args.subs_rate, len(lm.vocab), args.del_rate, args.ins_rate, protected=[lm.vocab['</s>']])

    inputs, targets = corrupted_provider.provide()

    for _, input, target in zip(range(args.nb_tokens), inputs, targets):
        in_word = lm.vocab.i2w(input.item())
        target_word = lm.vocab.i2w(target.item())
        print(in_word, target_word)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='location of the train corpus')
    parser.add_argument('--nb-tokens', type=int, default=10,
                        help='how many input-target pairs to show')

    parser.add_argument('--subs-rate', type=float, required=True,
                        help='what ratio of input tokens should be randomly')
    parser.add_argument('--del-rate', type=float, required=True,
                        help='what ratio of tokens should be removed')
    parser.add_argument('--ins-rate', type=float, required=True,
                        help='what ratio of tokens should be inserted')

    parser.add_argument('--load', type=str, required=True,
                        help='where to load a model from')
    args = parser.parse_args()

    main(args)
