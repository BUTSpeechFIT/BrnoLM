import argparse
import torch
from brnolm.language_models import language_model


def main(args):
    if args.force_cpu:
        lm = torch.load(args.lm, map_location='cpu')
    else:
        lm = torch.load(args.lm)
    language_model.torchscript_export(lm, args.frozen_lm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('lm')
    parser.add_argument('frozen_lm')
    args = parser.parse_args()

    main(args)
