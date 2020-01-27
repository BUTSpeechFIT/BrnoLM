#!/usr/bin/env python

import argparse
import math
import runtime_utils
import vocab

import torch
import analysis


def bows_to_ps(bows):
    uni_ps = bows.t() / bows.sum(dim=1)
    return uni_ps.t()
    

def bows_to_ent(bows):
    uni_ps = bows_to_ps(bows)
    entropies = analysis.categorical_entropy(uni_ps)

    avg_entropy = entropies @ bows.sum(dim=1) / bows.sum()
    return avg_entropy
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list')
    parser.add_argument('--vocab')
    parser.add_argument('--unk', default='<unk>')
    args = parser.parse_args()

    fns = runtime_utils.filenames_file_to_filenames(args.file_list)

    documents = []
    for fn in fns:
        with open(fn) as f:
            documents.append(f.read().split())
            
    with open(args.vocab) as f:
        vocab = vocab.vocab_from_kaldi_wordlist(f, args.unk)

    bows = torch.zeros(len(documents), len(vocab)).long()

    for doc_no, doc in enumerate(documents):
        for w in doc:
            bows[doc_no, vocab[w]] += 1

    avg_entropy = bows_to_ent(bows.float())
    print("{:.4f} {:.2f}".format(avg_entropy, 2**avg_entropy))

    bows_combined = bows.sum(dim=0, keepdim=True)  
    overall_entropy = bows_to_ent(bows_combined.float())
    print("{:.4f} {:.2f}".format(overall_entropy, 2**overall_entropy))
