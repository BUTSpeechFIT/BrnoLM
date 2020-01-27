#!/usr/bin/env python3
'''Migrates old LM from before proper brnolm package was introduced.

Build around the proposition of this SO answer:
https://stackoverflow.com/a/53327348/9703830

Uses a separate, monkey-patched pickle (`my_pickle`) for de-serialization
in order to ensure that the pure system pickle is ready to serialize the model.
'''

import argparse
import importlib
import pickle
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()

    my_pickle = load_module_extra('pickle')
    my_pickle.Unpickler = MyUnpickler

    lm = torch.load(args.source, map_location='cpu', pickle_module=my_pickle)
    torch.save(lm, args.target)


def load_module_extra(identifier):
    spec = importlib.util.find_spec(identifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class MyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module.startswith('language_models'):
            renamed_module = 'brnolm.' + module
        return super().find_class(renamed_module, name)


if __name__ == '__main__':
    main()
