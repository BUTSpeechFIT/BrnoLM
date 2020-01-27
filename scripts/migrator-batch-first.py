#!/usr/bin/env python3
'''Migrates old LM from before proper brnolm package was introduced.

Build around the proposition of this SO answer:
https://stackoverflow.com/a/53327348/9703830

Uses a separate, monkey-patched pickle (`my_pickle`) for de-serialization
in order to ensure that the pure system pickle is ready to serialize the model.
'''

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()

    lm = torch.load(args.source, map_location='cpu')
    lm.model.rnn.batch_first = True
    lm.model.batch_first = True
    torch.save(lm, args.target)


if __name__ == '__main__':
    main()
