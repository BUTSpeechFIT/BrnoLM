#!/usr/bin/env python

import pickle
import argparse

import plotting

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--color-bar", action='store_true')
    parser.add_argument("activations")
    args = parser.parse_args()

    with open(args.activations, 'rb') as f:
        activations = pickle.load(f)

    fig_titles = []
    for a in activations:
        nb_zeros = np.sum(a==0.0)
        fig_titles.append(", ".join([str(x) for x in [nb_zeros, np.mean(a), np.max(a)]]))

    color_setup = {
        'cmap': 'gnuplot',
        'vmin': 0,
        'vmax': 30,
        'colorbar': args.color_bar,
    }

    plotting.grid_plot(activations, lambda x:x, "activations", fig_titles, coloring=color_setup)
