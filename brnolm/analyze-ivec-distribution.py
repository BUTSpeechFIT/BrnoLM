#!/usr/bin/env python
import argparse
import io
import math
import sys
import torch
import numpy as np

import split_corpus_dataset
import ivec_appenders
import smm_ivec_extractor

from runtime_utils import filenames_file_to_filenames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list', required=True,
                        help="file with list of files analyze")
    parser.add_argument('--ivec-extractor', required=True,
                        help="iVector extractor to use")
    parser.add_argument('--output', required=True,
                        help="where to put the ivectors")
    args = parser.parse_args()
    print(args)

    print("loading SMM iVector extractor ...")
    with open(args.ivec_extractor, 'rb') as f:
        ivec_extractor = smm_ivec_extractor.load(f)
    print(ivec_extractor)

    documents = filenames_file_to_filenames(args.file_list)

    ivecs = []
    for doc in documents:
        with open(doc) as f:
            content = f.read()

        complete_ivec = ivec_extractor(content)
        ivecs.append(complete_ivec)

    ivecs = torch.stack(ivecs)
    print(ivecs)

    with open(args.output, 'w') as f:
        np.savetxt(f, ivecs.cpu().numpy()) 
