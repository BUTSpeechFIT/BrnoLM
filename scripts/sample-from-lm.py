#!/usr/bin/env python3

import argparse
import torch
from torch.distributions import Categorical

import sys


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def get_max(log_probs):
    return torch.max(log_probs, 0)[1]


def sample(log_probs, temperature):
    annealed_logits = log_probs / temperature
    dist = Categorical(logits=annealed_logits)
    return dist.sample()


def get_sampler(args):
    if args.sampler == 'max':
        return get_max
    elif args.sampler == 'sample':
        return lambda y: sample(y, args.temperature)
    else:
        raise ValueError(f"Unacceptable sampler {args.sampler}")


class NextIndexProducer:
    def __init__(self, lm, sampler, seed_text):
        self.lm = lm
        self.sampler = sampler
        self.inds_to_process = [lm.vocab[c] for c in seed_text]
        self.h = self.lm.model.init_hidden(1)

    def __call__(self):
        o, self.h = self.lm.model(torch.tensor(self.inds_to_process).unsqueeze(0), self.h)
        y = self.lm.decoder(o[0, -1, :]).squeeze().detach()

        sample = self.sampler(y).item()
        self.inds_to_process = [sample]

        return self.lm.vocab.i2w(sample)


class LineWriter:
    def __init__(self, f):
        self._f = f

    def write(self, string):
        self._f.write(string)

    def __enter__(self):
        self._put_delim()
        return self

    def __exit__(self, type, value, traceback):
        self._put_delim()
        self._f.write('\n')

    def _put_delim(self):
        self._f.write("'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sb')
    parser.add_argument('--sampler', choices=['max', 'sample'], default='sample')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('lm')
    parser.add_argument('seed_text')
    parser.add_argument('nb_tokens', nargs='?', type=int, default=10)
    args = parser.parse_args()

    lm = torch.load(args.lm, map_location="cpu")
    if args.sb:
        assert args.sb in lm.vocab

    index_producer = NextIndexProducer(lm, get_sampler(args), args.seed_text)

    sys.stdout = Unbuffered(sys.stdout)
    with LineWriter(sys.stdout) as writer:
        writer.write(args.seed_text)
        for i in range(args.nb_tokens):
            char = index_producer()

            if char == args.sb:
                char = '\n'

            writer.write(char)


if __name__ == '__main__':
    main()
