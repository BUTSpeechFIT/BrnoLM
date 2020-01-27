#!/usr/bin/env python

import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('first_scores', help='first set of scores to consider')
    parser.add_argument('first_weight', type=float, help='weight of the first set')
    parser.add_argument('second_scores', help='second set of scores to consider')
    parser.add_argument('second_weight', type=float, help='weight of the second set')
    args = parser.parse_args()

    sys.stderr.write("{}\n".format(sys.argv))

    with open(args.first_scores, 'r') as first_f, open(args.second_scores, 'r') as second_f:

        for first_line, second_line in zip(first_f, second_f):
            first_fields = first_line.split()
            second_fields = second_line.split()

            assert first_fields[0] == second_fields[0]

            first_s = float(first_fields[1])
            second_s = float(second_fields[1])

            combined_score = first_s * args.first_weight + second_s * args.second_weight
            print("{} {:.4f}".format(first_fields[0], combined_score))
