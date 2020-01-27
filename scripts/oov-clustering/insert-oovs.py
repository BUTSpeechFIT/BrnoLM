#!/usr/bin/env python

import argparse
import copy
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unk', default="<UNK>")
    parser.add_argument('--unk-oi', default="<UNK-OI>")
    parser.add_argument('--oov-list', required=True)
    args = parser.parse_args()

    with open(args.oov_list) as f:
        oovs = f.read().split()

    oov_counts = {oov: 0 for oov in oovs}

    for line in sys.stdin:
        words = line.split()
        oov_hidden = [args.unk if w in oovs else w for w in words]

        if oov_hidden == words:
            continue

        for i, w in enumerate(words):
            if w in oovs:
                path_line = copy.deepcopy(oov_hidden)
                path_line[i] = args.unk_oi
                path_line_str = " ".join(path_line)

                path_key = "{}:{}".format(w, oov_counts[w])
                oov_counts[w] += 1
                sys.stdout.write("{} {}\n".format(path_key, path_line_str))
