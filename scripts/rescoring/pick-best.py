#!/usr/bin/env python

import argparse
import sys


def read_latt(f):
    line = f.readline()
    while line == '\n':
        line = f.readline()

    if line == '':
        return None, None, None

    fields = line.strip().split('-')
    segment_id = '-'.join(fields[:-1])
    trans_id = fields[-1]

    content = ""
    line = f.readline()
    while line != '\n':
        content += line
        line = f.readline()

    return segment_id, trans_id, content


def read_pick(f):
    return tuple(f.readline().split())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pick_file')

    args = parser.parse_args()

    tot_printed = 0
    unserved = 0

    with open(args.pick_file, 'r') as p_f:
        segment, best_trans = read_pick(p_f)
        served = False

        while True:
            seg_id, trans_id, latt = read_latt(sys.stdin)
            if not seg_id:
                if p_f.readline() == '':  # both files ended
                    break
                else:
                    raise ValueError("Latts file (stdin) ended sooner than picks file")

            if seg_id != segment:
                if not served:
                    sys.stderr.write("Unserved picks of segment " + segment + ", wanted " + best_trans + "-th transcription \n")
                    unserved += 1

                segment, best_trans = read_pick(p_f)
                served = False

            if trans_id == best_trans:
                print(segment + "\n" + latt + "\n")
                tot_printed += 1
                served = True

    if unserved > 0:
        sys.stderr.write("ERROR: Unserved " + str(unserved) + " picks \n")
        sys.exit(1)
