#!/usr/bin/env python
import argparse
import sys

import torch

from brnolm.oov_clustering.embeddings_io import str_from_embedding
from brnolm.oov_clustering.embeddings_computation import tensor_from_words


def relevant_prefix(transcript, word_of_interest):
    first_oov_oi_loc = transcript.index(word_of_interest)
    if word_of_interest in transcript[first_oov_oi_loc+1:]:
        raise ValueError("there are multiple OOVs of interest!")

    return transcript[:first_oov_oi_loc]


BATCH_SIZE = 1


def emb_from_string(transcript, lm):
    prefix = relevant_prefix(transcript, args.unk_oi)
    prefix = ["</s>"] + prefix

    th_data = tensor_from_words(prefix, lm.vocab)
    h0 = lm.model.init_hidden(th_data.size(0))

    if not lm.model.batch_first:
        th_data = th_data.t()
    emb, h = lm.model(th_data, h0)
    out_emb = emb[0][-1].data

    return out_emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unk', default="<UNK>")
    parser.add_argument('--unk-oi', default="<UNK-OI>")
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

    for line_no, line in enumerate(sys.stdin):
        fields = line.split()
        key = fields[0]
        transcript = fields[1:]

        output = key
        if args.fwd_lm:
            fwd_emb = emb_from_string(transcript, fwd_lm)
            output += " " + str_from_embedding(fwd_emb)
        if args.bwd_lm:
            bwd_emb = emb_from_string(list(reversed(transcript)), bwd_lm)
            output += " " + str_from_embedding(bwd_emb)
        output += '\n'

        sys.stdout.write(output)
