#!/usr/bin/env python

import argparse
import math
import runtime_utils

import torch

import language_model
import smm_ivec_extractor


def bows_to_ps(bows):
    uni_ps = bows.t() / bows.sum(dim=1)
    return uni_ps.t()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list')
    parser.add_argument('--ivec-lm')
    parser.add_argument('--ivec-extractor')
    args = parser.parse_args()

    print("loading LM...")
    with open(args.ivec_lm, 'rb') as f:
        lm = language_model.load(f)
    lm.model.cuda()
    print(lm.model)

    print("loading SMM iVector extractor ...")
    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    print(ivec_extractor)

    fns = runtime_utils.filenames_file_to_filenames(args.file_list)
    documents = []
    for fn in fns:
        with open(fn) as f:
            documents.append(f.read().split())

    bows = torch.zeros(len(documents), len(lm.vocab)).long()
    for doc_no, doc in enumerate(documents):
        for w in doc:
            bows[doc_no, lm.vocab[w]] += 1
    unigram_ps = bows_to_ps(bows.float()).cuda()

    cross_entropies = []
    for doc_no, doc in enumerate(documents):
        text = " ".join(doc)
        ivec = ivec_extractor(text).cuda()
        qs = lm.model.ivec_to_logprobs(ivec).data
        cross_entropies.append(unigram_ps[doc_no] @ qs)

    cross_entropies = torch.FloatTensor(cross_entropies)
    avg_ce = -cross_entropies @ bows.float().sum(dim=1) / bows.sum()

    print("{:.4f} {:.2f}".format(avg_ce, math.exp(avg_ce)))
