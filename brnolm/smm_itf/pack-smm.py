#!/usr/bin/env python
import argparse
import os

import pickle

import sys
from smm import SMM
import utils

import .smm_ivec_extractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smm', required=True, help="path to a trained SMM model")
    parser.add_argument('--tokenizer', required=True, help="path to the tokenizer used for processing the data for training the SMM")
    parser.add_argument('--save', required=True, help="where to put the final SMM ivec extractor")
    args = parser.parse_args()

    model_f = os.path.realpath(args.smm)
    model, config = utils.load_model_and_config(model_f)

    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    ivec_extractor = smm_ivec_extractor.IvecExtractor(model, nb_iters=10, lr=config['eta'], tokenizer=tokenizer)

    with open(args.save, 'wb') as f:
        ivec_extractor.save(f)
