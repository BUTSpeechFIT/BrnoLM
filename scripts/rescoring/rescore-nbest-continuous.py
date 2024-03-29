#!/usr/bin/env python3

import argparse
import logging
import torch

from brnolm.rescoring.segment_scoring import SegmentScorer
from safe_gpu.safe_gpu import GPUOwner

import typing

import brnolm.kaldi_itf


def select_hidden_state_to_pass(hidden_states):
    return hidden_states['1']


def spk_sess(segment_name):
    return segment_name.split('-')[0].split('_')


def main(args):
    logging.info(args)

    logging.info("reading model...")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    lm = torch.load(args.model_from, map_location=device)

    lm.eval()

    curr_seg = ''
    segment_utts: typing.Dict[str, typing.Any] = {}

    custom_h0 = None
    nb_carry_overs = 0
    nb_new_hs = 0

    with open(args.in_filename) as in_f, open(args.out_filename, 'w') as out_f:
        scorer = SegmentScorer(lm, out_f)

        for line in in_f:
            fields = line.split()
            segment, trans_id = brnolm.kaldi_itf.split_nbest_key(fields[0])

            words = fields[1:]

            if not curr_seg:
                curr_seg = segment

            if segment != curr_seg:
                result = scorer.process_segment(curr_seg, segment_utts, custom_h0)
                if args.carry_over == 'always':
                    custom_h0 = select_hidden_state_to_pass(result.hidden_states)
                    nb_carry_overs += 1
                elif args.carry_over == 'speaker':
                    if spk_sess(segment) == spk_sess(curr_seg):
                        custom_h0 = select_hidden_state_to_pass(result.hidden_states)
                        nb_carry_overs += 1
                    else:
                        custom_h0 = None
                        nb_new_hs += 1
                elif args.carry_over == 'never':
                    custom_h0 = None
                    nb_new_hs += 1
                else:
                    raise ValueError(f'Unsupported carry over regime {args.carry_over}')
                for hyp_no, cost in result.scores.items():
                    out_f.write(f"{curr_seg}-{hyp_no} {cost}\n")

                curr_seg = segment
                segment_utts = {}

            segment_utts[trans_id] = words

        # Last segment:
        result = scorer.process_segment(curr_seg, segment_utts)
        for hyp_no, cost in result.scores.items():
            out_f.write(f"{curr_seg}-{hyp_no} {cost}\n")

    logging.info(f'Hidden state was carried over {nb_carry_overs} times and reset {nb_new_hs} times')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--carry-over', default='always', choices=['always', 'speaker', 'never'],
                        help='When to use the previous hidden state')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--character-lm', action='store_true',
                        help='Process strings by characters')
    parser.add_argument('--model-from', type=str, required=True,
                        help='where to load the model from')
    parser.add_argument('in_filename', help='second output of nbest-to-linear, textual')
    parser.add_argument('out_filename', help='where to put the LM scores')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.cuda:
        gpu_owner = GPUOwner()

    main(args)
