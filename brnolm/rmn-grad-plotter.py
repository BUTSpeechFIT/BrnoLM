#!/usr/bin/env python

import pickle
import argparse

import plotting

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--norms", action='store_true',
        help="print average norm of the gradient per layer")
    parser.add_argument("grads")
    args = parser.parse_args()

    with open(args.grads, 'rb') as f:
        grads = pickle.load(f)

    fig_titles = []
    if args.norms:
        for g in grads:
            norms = np.linalg.norm(g, ord=2, axis=1)
            fig_titles.append(", ".join([str(x) for x in [np.min(norms), np.mean(norms), np.max(norms)]]))

    coloring = {
        'vmin' : -5e-3,
        'vmax' : 5e-3,
        'cmap' : 'RdBu'
    }

    plotting.grid_plot(grads, lambda x:x, "Weighted grads", fig_titles, coloring)
