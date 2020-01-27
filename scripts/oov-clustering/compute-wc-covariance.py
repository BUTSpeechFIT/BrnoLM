#!/usr/bin/env python

import argparse
import sys

import numpy as np
from scipy.linalg import fractional_matrix_power

from brnolm.oov_clustering.embeddings_io import all_embs_by_key

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-cov', action='store_true')
    parser.add_argument(
        '--filter',
        help='file with words. If present, cov_wc will be computed exclusively from these'
    )
    args = parser.parse_args()

    if args.filter:
        with open(args.filter) as f:
            words_to_collect = f.read().split()
        shall_be_collected = lambda w: w in words_to_collect
    else:
        shall_be_collected = lambda _: True

    collection = all_embs_by_key(sys.stdin, shall_be_collected)
    sys.stderr.write(
        "INFO: Used total of {} words, {} unique\n".format(
            sum(block.shape[0] for block in collection.values()),
            len(collection)
        )
    )

    centered = []
    for w in collection:
        w_vectors = collection[w]
        mean = w_vectors.mean(axis=0)
        centered.append(w_vectors - mean)

    all_centered = np.concatenate(centered)
    covariance = np.cov(all_centered, rowvar=False)

    whitener = fractional_matrix_power(covariance, -0.5)
    np.savetxt(sys.stdout, whitener)

    if args.show_cov:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(covariance)
        plt.colorbar()
        plt.show()
