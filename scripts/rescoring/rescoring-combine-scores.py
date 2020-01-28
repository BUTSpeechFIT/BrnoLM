#!/usr/bin/env python

import argparse

import brnolm.kaldi_itf as kaldi_itf


def dict_argmin(dict):
    return min(dict, key=dict.get)


def write_best(scores, key, out_f):
    best = dict_argmin(scores)
    out_f.write(key + ' ' + best + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--ac-scale', type=float, default=1.0, help='weight of acoustic score')
    parser.add_argument('--gr-scale', type=float, required=True, help='weight of graph scores')
    parser.add_argument('--lm-scale', type=float, required=True, help='weight of rnnlm scores')
    parser.add_argument('acoustic_scores', help='file with acoustic scores')
    parser.add_argument('graph_scores', help='file with graph scores')
    parser.add_argument('rnnlm_scores', help='file with rnnlm scores')
    parser.add_argument('out_filename', help='where to put the best picked transcripts')
    args = parser.parse_args()

    print(args)

    curr_seg = None
    segment_utts_scores = {}

    with open(args.acoustic_scores, 'r') as ac_f, \
         open(args.graph_scores, 'r') as gr_f, \
         open(args.rnnlm_scores, 'r') as lm_f, \
         open(args.out_filename, 'w') as out_f:

        for ac_line, gr_line, lm_line in zip(ac_f, gr_f, lm_f):
            ac_fields = ac_line.split()
            gr_fields = gr_line.split()
            lm_fields = lm_line.split()

            assert ac_fields[0] == gr_fields[0] and gr_fields[0] == lm_fields[0]
            segment, trans_id = kaldi_itf.split_nbest_key(ac_fields[0])

            if not curr_seg:
                curr_seg = segment

            if segment != curr_seg:
                write_best(segment_utts_scores, curr_seg, out_f)

                curr_seg = segment
                segment_utts_scores = {}

            ac_s = float(ac_fields[1])
            gr_s = float(gr_fields[1])
            lm_s = float(lm_fields[1])

            segment_utts_scores[trans_id] = (
                args.ac_scale * ac_s +
                args.gr_scale * gr_s +
                args.lm_scale * lm_s
            )

        write_best(segment_utts_scores, curr_seg, out_f)
