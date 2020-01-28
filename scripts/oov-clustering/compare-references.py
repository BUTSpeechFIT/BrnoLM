#!/usr/bin/env python
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ref1')
    parser.add_argument('ref2')
    args = parser.parse_args()

    nb_00 = 0
    nb_01 = 0
    nb_10 = 0
    nb_11 = 0

    with open(args.ref1) as f1, open(args.ref2) as f2:
        for l1, l2 in zip(f1, f2):
            fields_1 = [int(float(r)) for r in l1.split()]
            fields_2 = [int(float(r)) for r in l2.split()]

            for r1, r2 in zip(fields_1, fields_2):
                if r1 == 0 and r2 == 0:
                    nb_00 += 1
                elif r1 == 0 and r2 == 1:
                    nb_01 += 1
                elif r1 == 1 and r2 == 0:
                    nb_10 += 1
                elif r1 == 1 and r2 == 1:
                    nb_11 += 1

    nb_total = nb_00 + nb_01 + nb_10 + nb_11

    print("{:.2f} {:.2f}".format(100.0*nb_00/nb_total, 100.0*nb_01/nb_total))
    print("{:.2f} {:.2f}".format(100.0*nb_10/nb_total, 100.0*nb_11/nb_total))
