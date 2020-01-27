#!/usr/bin/env python

import argparse
from os.path import commonprefix
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform

from brnolm.oov_clustering.embeddings_io import all_embs_by_key
from brnolm.oov_clustering.det import DETCurve


def extract_unique_scores(square_scores):
    return square_scores[np.triu_indices(square_scores.shape[0], k=0)]


def only_differ_in_suffix(a, b, suffix_maxlen=1):
    prefix = commonprefix([a, b])
    a_suffix = a[len(prefix):]
    b_suffix = a[len(prefix):]

    return max([len(a_suffix), len(b_suffix)]) <= suffix_maxlen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-det', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-3, help='to prevent log of zero')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--trials', required=True, help='file with word pairs to compare')
    parser.add_argument('--disregard-suffixes', action='store_true')
    parser.add_argument('--free-axis', action='store_true')
    parser.add_argument('--eer-line', action='store_true')
    parser.add_argument('--metric', default='inner_prod', choices=['inner_prod'])
    args = parser.parse_args()

    trial_pairs = []
    with open(args.trials) as f:
        for line in f:
            trial_pairs.append(tuple(line.split()))

    emb_collection = all_embs_by_key(sys.stdin, key_transform=lambda w: w.split(':')[0])

    score_tg = []
    for w in emb_collection:
        embs = emb_collection[w]
        similarities = embs @ embs.T
        score_tg.extend([(s, 1) for s in extract_unique_scores(similarities)])

    for a, b in trial_pairs:
        if a not in emb_collection or b not in emb_collection:
            continue

        if args.disregard_suffixes and only_differ_in_suffix(a, b):
            continue

        a_embs = emb_collection[a]
        b_embs = emb_collection[b]
        similarities = a_embs @ b_embs.T
        score_tg.extend([(s, 0) for s in similarities.flat])

    det = DETCurve(score_tg, args.baseline, max_det_points=200)
    sys.stdout.write(det.textual_report())
    if args.plot:
        det.plot(args.log_det, not args.free_axis, args.eer_line)
