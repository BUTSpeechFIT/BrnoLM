#!/usr/bin/env python
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', type=int, default=0, help='where to start numbering')
    parser.add_argument('file', help='where to collect characters from')
    args = parser.parse_args()

    bag_of_letters = set()
    with open(args.file) as f:
        for line in f:
            bag_of_letters = bag_of_letters | set(line)

    bag_of_letters = bag_of_letters - set('\n')

    for i, c in enumerate(bag_of_letters):
        sys.stdout.write("'{}' {}\n".format(c, i + args.offset))
