#!/usr/bin/env python
import argparse
import io
import math
import sys
import torch

import split_corpus_dataset
import ivec_appenders
import smm_ivec_extractor

from runtime_utils import filenames_file_to_filenames

class DummyDict:
    def __getitem__(self, index):
        return 0


def euclidean_distance(a, b):
    return (a-b).pow(2).sum(dim=-1).pow(0.5)


def length(a):
    return a.pow(2).sum(dim=-1).pow(0.5)


def euclidean_distance(a, b):
    return (a-b).pow(2).sum(dim=-1).pow(0.5)


def cosine_similarity(a, b):
    return (a*b).sum(dim=-1) / (length(a) * length(b))


def analyze_document(text, ivec_extractor):
    nb_words = len(text.split())

    if args.unroll_steps is None:
        unroll = args.unroll
    else:
        if nb_words % args.unroll_steps == 0:
            text = text.rsplit(maxsplit=1)[0]
            nb_words -= 1
        unroll = nb_words // args.unroll_steps

    complete_ivec = ivec_extractor(text)
    complete_ivec_len = length(complete_ivec)

    partial_ivecs = [
        ivec_extractor(" ".join(prefix)) for prefix in [
            text.split()[:l] for l in range(0, nb_words, unroll)
        ]
    ]

    if args.unroll_steps:
        partial_ivecs = torch.stack(partial_ivecs[:args.unroll_steps])

    if partial_ivecs.size(0) != args.unroll_steps:
        print(partial_ivecs.size(0))

    distances = euclidean_distance(partial_ivecs, complete_ivec)
    cos_sims = cosine_similarity(partial_ivecs, complete_ivec)

    return distances, complete_ivec_len, cos_sims


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unroll', type=int, default=35,
                        help="bptt equivalent. Applies only when --unroll-steps is not set.")
    parser.add_argument('--unroll-steps', type=int, 
                        help="for how many steps the text should unrolled. Overrides --unroll.")
    source_opt = parser.add_mutually_exclusive_group(required=True)
    source_opt.add_argument('--document', help="what to perform the analysis on")
    source_opt.add_argument('--file-list', help="file with list of files analyze")
    parser.add_argument('--ivec-extractor', required=True,
                        help="iVector extractor to use")
    args = parser.parse_args()
    print(args)

    print("loading SMM iVector extractor ...")
    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    print(ivec_extractor)

    if args.document:
        with open(args.document) as f:
            content = f.read()
        distances = analyze_document(content, ivec_extractor)

        print(distances)

    elif args.file_list:
        if not args.unroll_steps:
            raise ValueError("When analyzing a filelist, --unroll-steps HAS to be specified.")

        documents = filenames_file_to_filenames(args.file_list)

        distances = []
        ci_lens = []
        cos_sims = []
        nb_failed = 0
        for doc in documents:
            with open(doc) as f:
                content = f.read()

            try:
                distance, ci_len, cos_sim = analyze_document(content, ivec_extractor)
                distances.append(distance)
                ci_lens.append(ci_len)
                cos_sims.append(cos_sim)
            except ValueError:
                nb_failed += 1

        if nb_failed > 0:
            sys.stderr.write("Failed analyzing {} documents, because they are too short.\n".format(nb_failed))

        distances = torch.stack(distances)
        ci_lens = torch.stack(ci_lens)
        cos_sims = torch.stack(cos_sims)

        print(torch.stack([distances.min(dim=0)[0], distances.mean(dim=0), distances.max(dim=0)[0], distances.var(dim=0)]).t())

        distances_normed = distances / ci_lens
        print(torch.stack([
            distances_normed.min(dim=0)[0], distances_normed.mean(dim=0),
            distances_normed.max(dim=0)[0], distances_normed.var(dim=0)
        ]).t())

        print(torch.stack([cos_sims.min(dim=0)[0], cos_sims.mean(dim=0), cos_sims.max(dim=0)[0], cos_sims.var(dim=0)]).t())
