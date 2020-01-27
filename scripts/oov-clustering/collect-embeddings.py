#!/usr/bin/env python

import argparse
import sys

import torch

from brnolm.oov_clustering.embeddings_io import str_from_embedding
from brnolm.oov_clustering.embeddings_computation import tensor_from_words


def embs_from_words(words, lm):
    words = ["</s>"] + words
    th_data = tensor_from_words(words, lm.vocab)[:, :-1]
    h0 = lm.model.init_hidden(th_data.size(0))

    if not lm.model.batch_first:
        th_data = th_data.t()
    emb, h = lm.model(th_data, h0)
    return [an_embedding[0].detach() for an_embedding in emb]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fwd-lm')
    parser.add_argument('--bwd-lm')
    args = parser.parse_args()

    if not args.fwd_lm and not args.bwd_lm:
        sys.stderr.write("At least one of '--fwd-lm' and '--bwd-lm' needs to be specified\n")
        sys.exit(1)

    if args.fwd_lm:
        fwd_lm = torch.load(args.fwd_lm, map_location=lambda storage, location: storage)
        fwd_lm.eval()
    if args.bwd_lm:
        bwd_lm = torch.load(args.bwd_lm, map_location=lambda storage, location: storage)
        bwd_lm.eval()

    vocabulary = fwd_lm.vocab if args.fwd_lm else bwd_lm.vocab

    for line in sys.stdin:
        words = line.split()

        data_cols = [words]

        if args.fwd_lm:
            fwd_embs = embs_from_words(words, fwd_lm)
            fwd_embs_strs = [str_from_embedding(emb) for emb in fwd_embs]
            data_cols.append(fwd_embs_strs)

        if args.bwd_lm:
            bwd_embs = reversed(list(embs_from_words(list(reversed(words)), bwd_lm)))
            bwd_embs_strs = [str_from_embedding(emb) for emb in bwd_embs]
            data_cols.append(bwd_embs_strs)

        for data_row in zip(*data_cols):
            elem_strs = ["{}".format(elem) for elem in data_row]
            sys.stdout.write(" ".join(elem_strs) + "\n")
