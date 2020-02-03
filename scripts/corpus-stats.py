#!/usr/bin/env python

import argparse
import torch
import sys

from brnolm.data_pipeline.reading import tokens_from_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--characters', action='store_true',
                        help='work on character level, whitespace is significant')
    parser.add_argument('--lm', type=str, required=True,
                        help='where to load a model from')
    parser.add_argument('train', type=str,
                        help='location of the train corpus')
    args = parser.parse_args()

    lm = torch.load(args.lm, map_location='cpu')

    tokenize_regime = 'words'
    if args.characters:
        tokenize_regime = 'chars'

    train_ids = tokens_from_fn(args.train, lm.vocab, randomize=False, regime=tokenize_regime)
    sys.stdout.write('Vocabulary size: {}\n'.format(len(lm.vocab)))

    nb_tokens = len(train_ids)
    sys.stdout.write('Nb tokens: {}\n'.format(nb_tokens))

    oov_mask = train_ids == lm.vocab.unk_ind
    nb_oovs = oov_mask.sum()

    sys.stdout.write('Nb oovs: {} ({:.2f} %)\n'.format(nb_oovs, 100.0 * nb_oovs/nb_tokens))
