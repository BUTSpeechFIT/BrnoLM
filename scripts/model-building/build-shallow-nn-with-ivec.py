#!/usr/bin/env python
import argparse
import torch

from brnolm.language_models import language_model, vocab, ffnn_models
from brnolm.language_models.decoders import FullSoftmaxDecoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FFNN Language Model')
    parser.add_argument('--wordlist', type=str, required=True,
                        help='word -> int map; Kaldi style "words.txt"')
    parser.add_argument('--unk', type=str, default="<unk>",
                        help='expected form of "unk" word. Most likely a <UNK> or <unk>')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--ivec-dim', type=int, required=True,
                        help='ivector dimensionality')
    parser.add_argument('--hist-len', type=int, default=2,
                        help='number of input words. If n-grams are being modelled, then (n-1)')
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
        vocabulary = vocab.vocab_from_kaldi_wordlist(f, args.unk)

    print("building model...")

    model = ffnn_models.BengioModelIvecInput(
        len(vocabulary), args.emsize, args.hist_len,
        args.nhid, args.dropout, args.ivec_dim
    )

    decoder = FullSoftmaxDecoder(args.nhid, len(vocabulary))

    lm = language_model.LanguageModel(model, decoder, vocabulary)
    torch.save(lm, args.save)
