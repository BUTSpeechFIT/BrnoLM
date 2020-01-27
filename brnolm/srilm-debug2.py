#!/usr/bin/env python
import argparse
import math
import sys

import torch


BATCH_SIZE = 1


def per_word_logprobs(words, lm):
    words_tensor = torch.tensor([lm.vocab[w] for w in words], requires_grad=False).view(1, -1)

    x = words_tensor[:, :-1]
    t = words_tensor[:, 1:]

    h0 = lm.model.init_hidden(x.size(0))
    if not lm.model.batch_first:
        x = x.t()
        t = t.t()

    with torch.no_grad():
        out_embs, h = lm.model(x, h0)
        log_probs = -lm.decoder.neg_log_prob_raw(out_embs, t)

    return log_probs.view(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm', required=True)
    parser.add_argument('--sb', default='</s>')
    parser.add_argument('--add-boundaries', action='store_true')
    args = parser.parse_args()

    lm = torch.load(args.lm, map_location=lambda storage, location: storage)
    lm.eval()

    nb_sentences = 0
    nb_words = 0
    first_line = True
    total_logprob = 0.0
    sentence_boundary_logprob = 0.0
    for line_no, line in enumerate(sys.stdin):
        words = line.split()

        if len(words) <= 1:
            sys.stderr.write('Skipping line {} due to only containing "{}".\n'.format(line_no, words))
            continue

        if args.add_boundaries:
            if words[-1] == args.sb:
                sys.stderr.write('Skipping line {}, contains sentence boundary, despite --add-boundaries. "{}".'.format(line_no, args.sb))
                continue
            else:
                words.append(args.sb)
        else:
            if words[-1] != args.sb:
                sys.stderr.write('Skipping line {}, lacks sentence boundary "{}".'.format(line_no, args.sb))
                continue

        log_probs = per_word_logprobs([args.sb] + words, lm)

        if first_line:
            first_line = False
        else:
            sys.stdout.write('\n')

        sys.stdout.write(line)
        for w, log_p in zip(words, log_probs):
            word_field = "p( {} | ...)".format(w)
            sys.stdout.write("\t{}\t= [1gram] {:.3f}\n".format(word_field, log_p.item(), log_p.exp().item()))

        nb_sentences += 1
        nb_words += len(words)
        total_logprob += log_probs.sum().item()
        sentence_boundary_logprob += log_probs[-1]

    nb_words -= nb_sentences
    sys.stdout.write('{} sentences, {} words, 0 OOVs\n'.format(nb_sentences, nb_words))
    sys.stdout.write('0 zeroprobs, logprob= {:.3f} ppl= {:.2f} ppl1= {:.2f}\n'.format(
        total_logprob, math.exp(-total_logprob/(nb_words+nb_sentences)),
        math.exp(-(total_logprob-sentence_boundary_logprob)/nb_words)
    ))


if __name__ == '__main__':
    main()
