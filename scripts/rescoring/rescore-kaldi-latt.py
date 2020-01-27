#!/usr/bin/env python

import argparse
import torch

import brnolm.language_models.vocab as vocab

import typing

import brnolm.kaldi_itf


def seqs_to_tensor(seqs):
    batch_size = len(seqs)
    maxlen = max([len(seq) for seq in seqs])

    ids = torch.LongTensor(batch_size, maxlen).zero_()
    for seq_n, seq in enumerate(seqs):
        for word_n, word in enumerate(seq):
            ids[seq_n, word_n] = word

    return ids


def dict_to_list(utts_map):
    list_of_lists = []
    rev_map = {}
    for key in utts_map:
        rev_map[len(list_of_lists)] = key
        list_of_lists.append(utts_map[key])

    return list_of_lists, rev_map


def translate_latt_to_model(word_ids, latt_vocab, model_vocab, mode='words'):
    words = [latt_vocab.i2w(i) for i in word_ids]
    if mode == 'words':
        return tokens_to_pythlm(words, model_vocab)
    elif mode == 'chars':
        sentence = " ".join(words)
        return tokens_to_pythlm(list(sentence), model_vocab)
    else:
        raise ValueError('Got unexpected mode "{}"'.format(mode))


def pick_ys(y, seq_x):
    seqs_ys = []
    for seq_n, seq in enumerate(seq_x):
        seq_ys = [1.0]  # hard 1.0 for the 'sure' <s>
        for w_n, w in enumerate(seq[1:]):  # skipping the initial element ^^^
            seq_ys.append(y[w_n, seq_n, w])
        seqs_ys.append(seq_ys)

    return seqs_ys


def seqs_logprob(seqs, lm):
    ''' Sequence as a list of integers
    '''
    data = seqs_to_tensor(seqs)
    batch_size = data.size(0)

    if not lm.model.batch_first:
        data = data.t().contiguous()

    if next(lm.model.parameters()).is_cuda:
        data = data.cuda()

    X = data
    h0 = lm.model.init_hidden(batch_size)

    o, _ = lm.model(X, h0)
    y = lm.decoder(o)
    y = y.detach()  # extract the Tensor out of the Variable

    word_log_scores = pick_ys(y, seqs)
    seq_log_scores = [sum(seq) for seq in word_log_scores]

    return seq_log_scores


def tokens_to_pythlm(toks, vocab):
    return [vocab.w2i('<s>')] + [vocab.w2i(tok) for tok in toks] + [vocab.w2i("</s>")]


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

    print(args)

    mode = 'chars' if args.character_lm else 'words'

    print("reading lattice vocab...")
    with open(args.latt_vocab, 'r') as f:
        latt_vocab = vocab.vocab_from_kaldi_wordlist(f, unk_word=args.latt_unk)

    print("reading model...")
    lm = torch.load(args.model_from, map_location='cpu')
    if args.cuda:
        lm.model.cuda()
    lm.model.eval()

    print("scoring...")
    curr_seg = ''
    segment_utts: typing.Dict[str, typing.Any] = {}

    with open(args.in_filename) as in_f, open(args.out_filename, 'w') as out_f:
        for line in in_f:
            fields = line.split()
            segment, trans_id = brnolm.kaldi_itf.split_nbest_key(fields[0])

            word_ids = [int(wi) for wi in fields[1:]]
            ids = translate_latt_to_model(word_ids, latt_vocab, lm.vocab, mode)

            if not curr_seg:
                curr_seg = segment

            if segment != curr_seg:
                X, rev_map = dict_to_list(segment_utts)  # reform the word sequences
                y = seqs_logprob(X, lm)  # score

                # write
                for i, log_p in enumerate(y):
                    out_f.write(curr_seg + '-' + rev_map[i] + ' ' + str(-log_p.item()) + '\n')

                curr_seg = segment
                segment_utts = {}

            segment_utts[trans_id] = ids

        # Last segment:
        X, rev_map = dict_to_list(segment_utts)  # reform the word sequences
        y = seqs_logprob(X, lm)  # score

        # write
        for i, log_p in enumerate(y):
            out_f.write(curr_seg + '-' + rev_map[i] + ' ' + str(-log_p.item()) + '\n')


if __name__ == '__main__':
    main()
