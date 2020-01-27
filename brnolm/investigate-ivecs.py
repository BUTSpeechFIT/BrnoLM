#!/usr/bin/env python
import argparse
import torch

import smm_ivec_extractor

from runtime_utils import init_seeds, filenames_file_to_filenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--filelist', type=str, required=True,
                        help='file with paths to documents')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--ivec-extractor', type=str, required=True,
                        help='where to load a ivector extractor from')
    args = parser.parse_args()
    print(args)

    init_seeds(args.seed, args.cuda)

    print("loading SMM iVector extractor ...")
    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    print(ivec_extractor)

    print("reading data...")
    filenames = filenames_file_to_filenames(args.filelist)
    texts = []
    for fn in filenames:
        with open(fn) as f:
            texts.append(f.read())

    print("computing iVectors...")
    ivecs = [ivec_extractor(t) for t in texts]
    ivecs = torch.stack(ivecs)

    print("Elements mean:\t", ivecs.mean())
    print("Elements var:\t", ivecs.var())
    
    sq_magnitudes = ivecs.pow(2).sum(1)
    print("Sq magn mean:\t", sq_magnitudes.mean())
    print("Sq magn var:\t", sq_magnitudes.var())
