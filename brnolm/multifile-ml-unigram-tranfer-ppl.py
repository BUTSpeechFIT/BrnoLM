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

def documents_from_fn(fn_filelist):
    fns = runtime_utils.filenames_file_to_filenames(fn_filelist)
    documents = []
    for fn in fns:
        with open(fn) as f:
            documents.append(f.read().split())

    return documents

def bow_from_documents(documents, vocab):
    bows = torch.zeros(len(documents), len(vocab)).long()
    for doc_no, doc in enumerate(documents):
        for w in doc:
            bows[doc_no, vocab[w]] += 1

    return bows
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-list')
    parser.add_argument('--file-list')
    parser.add_argument('--vocab')
    parser.add_argument('--unk', default='<unk>')
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = vocab.vocab_from_kaldi_wordlist(f, args.unk)

    documents = documents_from_fn(args.source_list)
    bows = bow_from_documents(documents, vocab).float()
    unigram_ps = bows_to_ps(bows.sum(dim=0, keepdim=True)).squeeze()

    test_documents = documents_from_fn(args.file_list)
    test_bows = bow_from_documents(test_documents, vocab).float()
    test_unigrams = bows_to_ps(test_bows)

    print(unigram_ps.size())
    print(test_unigrams.size())

    cross_entropies = analysis.categorical_cross_entropy(test_unigrams, unigram_ps)
    # print(cross_entropies)
    
    test_lengths = test_bows.sum(dim=1)
    avg_entropy =  cross_entropies @ test_lengths / test_bows.sum()
    print("{:.4f} {:.2f}".format(avg_entropy, 2**avg_entropy))
