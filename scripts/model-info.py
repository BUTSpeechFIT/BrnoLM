#!/usr/bin/env python3
import argparse
import torch

from brnolm.runtime.model_statistics import ModelStatistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    args = parser.parse_args()

    lm = torch.load(args.model_path, map_location='cpu')
    print(ModelStatistics(lm))


if __name__ == '__main__':
    main()
