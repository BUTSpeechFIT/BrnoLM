#!/usr/bin/env python

import argparse
from brnolm.oov_clustering.det import DETCurve
from typing import List, Tuple
import pickle
import random
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True,
                        help='reference matrix, text, triangular')
    parser.add_argument('--selection', help='file with candidates to consider')
    parser.add_argument('--posteriors', help='file with per-candidate posteriors')
    parser.add_argument('--posteriors-percentage', type=float, default=0.1,
                        help='file with per-candidate posteriors')
    parser.add_argument('--scores', required=True,
                        help='reference matrix, text, triangular')
    parser.add_argument('--sampling-rate', type=float, default=0.1,
                        help='reference matrix, text, triangular')
    parser.add_argument('--det-file', required=True,
                        help='where to put the pickled DETCurve object')
    args = parser.parse_args()

    if args.posteriors and not args.selection:
        posteriors = []
        with open(args.posteriors) as f:
            for line in f:
                fields = line.split()
                posteriors.append(float(fields[1]))
        sorted_posteriors = sorted(posteriors, reverse=True)
        threshold = sorted_posteriors[int(len(posteriors)*args.posteriors_percentage)]
        index_is_interesting = lambda x: posteriors[x] > threshold
    elif args.posteriors and args.selection:
        all_candidates = []
        with open(args.posteriors) as f:
            for line in f:
                fields = line.split()
                all_candidates.append(fields[0])
        with open(args.selection) as f:
            selected = f.read().split()
            indexes_of_interesting = [all_candidates.index(s) for s in selected]
        index_is_interesting = lambda x: x in indexes_of_interesting
    elif not args.posteriors and args.selection:
        sys.stderr.write("Once selection is given, posteriors are necessary\n")
        sys.exit(1)
    else:
        index_is_interesting = lambda x: True

    score_tg: List[Tuple[float, float]] = []
    with open(args.ref) as ref_f, open(args.scores) as scores_f:
        for i, (ref_line, score_line) in enumerate(zip(ref_f, scores_f)):
            if not index_is_interesting(i):
                continue

            ref_fields = [float(x) for x in ref_line.split()]
            score_fields = [float(x) for x in score_line.split()]

            line_score_tg = list(zip(score_fields, ref_fields))
            if args.posteriors:
                line_score_tg = [
                    stg for j, stg in enumerate(line_score_tg) if index_is_interesting(j)
                ]

            score_tg.extend(random.sample(
                line_score_tg,
                int(len(line_score_tg)*args.sampling_rate)
            ))

    det = DETCurve(score_tg, baseline=True, max_det_points=500)

    with open(args.det_file, 'wb') as f:
        pickle.dump(det, f)
