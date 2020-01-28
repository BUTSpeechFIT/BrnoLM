#!/usr/bin/env python
import argparse
import sys


def levenshtein_distance(s1, s2):
    ''' taken from stackoverflow
    '''
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexicon', help='Compute edit dist. in a phonetic space given by the LEXICON')
    args = parser.parse_args()

    lexicon = {}
    if args.lexicon:
        with open(args.lexicon) as f:
            for line in f:
                fields = line.split()
                lexicon[fields[0]] = fields[1:]

    words = sys.stdin.read().split()
    unique_pairs = [(words[i], words[j]) for i in range(len(words)) for j in range(i)]

    if args.lexicon:
        pair_distances = sorted(
            [(a, b, levenshtein_distance(lexicon[a], lexicon[b])) for a, b in unique_pairs],
            key=lambda pair_with_dist: pair_with_dist[2]
        )
    else:
        pair_distances = sorted(
            [(a, b, levenshtein_distance(a, b)) for a, b in unique_pairs],
            key=lambda pair_with_dist: pair_with_dist[2]
        )

    for a, b, d in pair_distances:
        sys.stdout.write("{} {} {}\n".format(a, b, d))
