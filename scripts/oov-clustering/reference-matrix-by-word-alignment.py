#!/usr/bin/env python

import argparse
import sys

import numpy as np

from brnolm.oov_clustering.oov_alignment_lib import align, extract_mismatch
from brnolm.oov_clustering.oov_alignment_lib import find_in_mismatches, number_of_errors

from typing import Dict, List, Tuple


def parse_oov_id(oov_id):
    return tuple(oov_id.split('_'))


def intersection(a, b):
    return list(set(a) & set(b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-references', required=True)
    parser.add_argument('--oov-list', required=True)
    parser.add_argument('--reference-file', required=True)
    args = parser.parse_args()

    with open(args.oov_list) as f:
        oov_list = f.read().split()

    np.set_printoptions(threshold=2000, linewidth=np.inf)

    total_nb_errors = 0
    total_ref_len = 0

    oov_hits: Dict[int, List[Tuple[str, List[str]]]] = {}

    references = {}
    with open(args.text_references) as f:
        for line in f:
            fields = line.split()
            references[fields[0]] = fields[1:]

    candidate_possible_words = []
    for line in sys.stdin:
        fields = line.split()
        _, utt_id, _, _, _ = parse_oov_id(fields[0])

        candidate_line = fields[1:]
        reference_line = references[utt_id]
        alignment = align(reference_line, candidate_line)
        mismatches = extract_mismatch(alignment)
        oov_mismatch = find_in_mismatches(mismatches, "<UNK-OI>")

        total_ref_len += len(reference_line)
        total_nb_errors += number_of_errors(mismatches)
        matching_oovs = intersection(oov_list, oov_mismatch[0])

        if len(matching_oovs) in oov_hits:
            oov_hits[len(matching_oovs)].append((utt_id, matching_oovs))
        else:
            oov_hits[len(matching_oovs)] = [(utt_id, matching_oovs)]

        candidate_possible_words.append(oov_mismatch[0])
        print(fields[0], oov_mismatch[0], '--', oov_mismatch[1])

    with open(args.reference_file, 'w') as f:
        for i, c1 in enumerate(candidate_possible_words):
            intersecting = []
            for c2 in candidate_possible_words[i+1:]:
                intersecting.append("1" if len(intersection(c1, c2)) > 0 else "0")

            f.write(" ".join(intersecting) + "\n")

    wer_fmt = "Total WER on candidate paths: {:.2f} % ({} / {})\n"
    sys.stderr.write(wer_fmt.format(100.0*total_nb_errors/total_ref_len, total_nb_errors, total_ref_len))

    oov_hit_fmt = "Total number of candidates where the hypothesised OOV may match an actual OOV : {:.2f} % ({} / {})\n"
    total_nb_candidates = sum(len(k) for k in oov_hits.values())
    total_nb_oov_hits = sum(len(oov_hits[k]) for k in oov_hits if k > 0)
    sys.stderr.write(oov_hit_fmt.format(100.0*total_nb_oov_hits/total_nb_candidates, total_nb_oov_hits, total_nb_candidates))
    for k in oov_hits:
        nb_candidates = len(oov_hits[k])
        percentage_candidates = 100.0 * nb_candidates/total_nb_candidates
        nb_unique_candidates = len(set((hit[0], tuple(hit[1])) for hit in oov_hits[k]))
        sys.stderr.write("{}: {} {:.2f}% {}\n".format(k, nb_candidates, percentage_candidates, nb_unique_candidates))
