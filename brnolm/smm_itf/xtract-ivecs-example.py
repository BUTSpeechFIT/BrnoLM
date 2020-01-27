#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : KarelB
# e-mail : ibenes AT fit.vutbr.cz

import argparse
import torch

import .smm_ivec_extractor


def bow_from_sentence(sentence, vocab):
    bow = torch.zeros((1, len(vocab))).float()

    for w in sentence.split():
        bow[0, vocab[w]] += 1.0

    return bow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete-smm", required=True, help="path to a complete SMM model file")
    parser.add_argument('--mkl', default=1, type=int, help='number of MKL threads')

    args = parser.parse_args()

    torch.set_num_threads(args.mkl)
    torch.manual_seed(0)

    with open(args.complete_smm, 'rb') as f:
        ivec_xtractor = smm_ivec_extractor.load(f)

    s_biology1 = "whale rat elephant hippopotamus bee zoology mammals"
    s_biology2 = "flower cat dog insect insect"
    s_buildings1 = "bridge tower house chimney"
    s_buildings2 = "castle factory architecture wall"

    sentences = [s_biology1, s_biology2, s_buildings1, s_buildings2]
    ivecs = [ivec_xtractor(s) for s in sentences]
    ivecs = torch.stack(ivecs)

    print(ivecs @ ivecs.t())
