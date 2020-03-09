#!/usr/bin/env python

import argparse
import logging
import torch

import brnolm.language_models.vocab as vocab

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


class SegmentScorer:
    def __init__(self, lm, out_f, max_softmaxes=2000):
        self.lm = lm
        self.out_f = out_f
        self.max_softmaxes = max_softmaxes

    def process_segment(self, seg_name, seg_hyps):
        nb_hyps = len(seg_hyps)
        min_len = min(len(hyp) for hyp in seg_hyps.values())
        max_len = max(len(hyp) for hyp in seg_hyps.values())
        total_len = sum(len(hyp) for hyp in seg_hyps.values())
        nb_oovs = sum(sum(token == self.lm.vocab.unk_word for token in hyp) for hyp in seg_hyps.values())
        logging.info(f"{seg_name}: {nb_hyps} hypotheses, min/max/avg length {min_len}/{max_len}/{total_len/nb_hyps:.1f} tokens, # OOVs {nb_oovs}")

        X, rev_map = self.dict_to_list(seg_hyps)  # reform the word sequences
        y = self.get_scores(X)

        for i, log_p in enumerate(y):
            self.out_f.write(f"{seg_name}-{rev_map[i]} {str(log_p)}\n")

    def dict_to_list(self, utts_map):
        list_of_lists = []
        rev_map = {}
        for key in utts_map:
            rev_map[len(list_of_lists)] = key
            list_of_lists.append(utts_map[key])

        return list_of_lists, rev_map

    def get_scores(self, hyps):
        work_left = [hyps]
        ys = []

        while work_left:
            batch = work_left.pop(0)
            try:
                if len(batch) * max(len(s) for s in batch) > self.max_softmaxes:
                    raise RuntimeError("Preemptive, batch is {len(batch)}x{max(len(s) for s in batch)}")
                this_batch_ys = self.lm.batch_nll(batch, prefix='</s>')
                ys.extend(this_batch_ys)
            except RuntimeError as e:
                cuda_memory_error = 'CUDA out of memory' in str(e)
                cpu_memory_error = "can't allocate memory" in str(e)
                preemtive_memory_error = "Preemptive" in str(e)
                assert cuda_memory_error or cpu_memory_error or preemtive_memory_error
                midpoint = len(batch) // 2
                assert midpoint > 0
                first, second = batch[:midpoint], batch[midpoint:]
                work_left.insert(0, second)
                work_left.insert(0, first)
        return ys


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
                scorer.process_segment(curr_seg, segment_utts)

                curr_seg = segment
                segment_utts = {}

            segment_utts[trans_id] = ids

        # Last segment:
        scorer.process_segment(curr_seg, segment_utts)


if __name__ == '__main__':
    main()
