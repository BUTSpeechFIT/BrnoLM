#!/usr/bin/env python

import argparse
import logging
import torch

import brnolm.language_models.vocab as vocab
from brnolm.rescoring.segment_scoring import SegmentScorer

import typing

import brnolm.kaldi_itf


def translate_latt_to_model(word_ids, latt_vocab, model_vocab, mode='words'):
    words = [latt_vocab.i2w(i) for i in word_ids]
    if mode == 'words':
        return words + ['</s>']
    elif mode == 'chars':
        chars = list(" ".join(words))
        return chars + ['</s>']
    else:
        raise ValueError('Got unexpected mode "{}"'.format(mode))


def select_hidden_state_to_pass(hidden_states):
    return hidden_states['1']


def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--latt-vocab', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
    parser.add_argument('--latt-unk', type=str, default='<unk>',
                        help='unk symbol used in the lattice')
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
    logging.info(args)

    mode = 'chars' if args.character_lm else 'words'

    logging.info("reading lattice vocab...")
    with open(args.latt_vocab, 'r') as f:
        latt_vocab = vocab.vocab_from_kaldi_wordlist(f, unk_word=args.latt_unk)

    logging.info("reading model...")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    lm = torch.load(args.model_from, map_location=device)

    lm.eval()

    curr_seg = ''
    segment_utts: typing.Dict[str, typing.Any] = {}

    custom_h0 = None

    with open(args.in_filename) as in_f, open(args.out_filename, 'w') as out_f:
        scorer = SegmentScorer(lm, out_f)

        for line in in_f:
            fields = line.split()
            segment, trans_id = brnolm.kaldi_itf.split_nbest_key(fields[0])

            word_ids = [int(wi) for wi in fields[1:]]
            ids = translate_latt_to_model(word_ids, latt_vocab, lm.vocab, mode)

            if not curr_seg:
                curr_seg = segment

            if segment != curr_seg:
                result = scorer.process_segment(curr_seg, segment_utts, custom_h0)
                custom_h0 = select_hidden_state_to_pass(result.hidden_states)
                for hyp_no, cost in result.scores.items():
                    out_f.write(f"{curr_seg}-{hyp_no} {cost}\n")

                curr_seg = segment
                segment_utts = {}

            segment_utts[trans_id] = ids

        # Last segment:
        result = scorer.process_segment(curr_seg, segment_utts)
        for hyp_no, cost in result.scores.items():
            out_f.write(f"{curr_seg}-{hyp_no} {cost}\n")


if __name__ == '__main__':
    main()
