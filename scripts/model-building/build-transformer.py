#!/usr/bin/env python

import argparse
import torch

from brnolm.language_models import transformer, vocab, language_model
from brnolm.language_models.decoders import FullSoftmaxDecoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
    parser.add_argument('--wordlist', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
    parser.add_argument('--quoted-wordlist', action='store_true',
                        help='assume the words are quoted (with a single quote)')
    parser.add_argument('--unk', type=str, default="<unk>",
                        help='expected form of "unk" word. Most likely a <UNK> or <unk>')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nheads', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--save', type=str, required=True,
                        help='path to save the final model')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    print("loading vocabulary...")
    with open(args.wordlist, 'r') as f:
        if args.quoted_wordlist:
            vocabulary = vocab.quoted_vocab_from_kaldi_wordlist(f, args.unk)
        else:
            vocabulary = vocab.vocab_from_kaldi_wordlist(f, args.unk)

    if not vocabulary.is_continuous():
        raise ValueError("Vocabulary is not continuous, missing indexes {}".format(vocabulary.missing_indexes()))

    print("building model...")

    model = transformer.TransformerLM(
        len(vocabulary),
        args.nheads, args.nhid, args.nlayers,
        args.dropout
    )

    decoder = FullSoftmaxDecoder(args.nhid, len(vocabulary))

    lm = language_model.LanguageModel(model, decoder, vocabulary)
    torch.save(lm, args.save)
