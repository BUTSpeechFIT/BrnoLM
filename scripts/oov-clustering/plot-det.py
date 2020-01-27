#!/usr/bin/env python

import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-det', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--free-axis', action='store_true')
    parser.add_argument('--eer-line', action='store_true')
    parser.add_argument('file', help="where is the pickled DETCurve")
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        det = pickle.load(f)

    det.plot(args.log_det, not args.free_axis, args.eer_line, filename=None)
