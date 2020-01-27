#!/usr/bin/env python

import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--vocab', action='store_true', help='print out the full vocab')
    parser.add_argument('load', help='where to load a model from')
    args = parser.parse_args()

    lm = torch.load(args.load, map_location='cpu')
    print(lm.model)
    print("Vocab len:", len(lm.vocab))
    if args.vocab:
        print([c for c in lm.vocab.w2i_])
