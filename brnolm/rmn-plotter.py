#!/usr/bin/env python

import torch
import argparse
import plotting
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")

    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = torch.load(f)

    cs_numpied = [c.weight.data.numpy() for c in model._cs]
    fig_titles = []
    for c in cs_numpied:
        norms = np.linalg.norm(c, ord=2, axis=1)
        fig_titles.append(", ".join([str(x) for x in [np.min(norms), np.mean(norms), np.max(norms)]]))
    plotting.grid_plot(cs_numpied, lambda x: x, "C-Weights", fig_titles)

    plotting.grid_plot(model._cs, lambda c: c.bias.data.numpy()[...,None], "Biases")

    ps_numpied = [p.weight.data.numpy() for p in model._ps]
    fig_titles = []
    for p in ps_numpied:
        norms = np.linalg.norm(p, ord=2, axis=1)
        fig_titles.append(", ".join([str(x) for x in [np.min(norms), np.mean(norms), np.max(norms)]]))
    plotting.grid_plot(ps_numpied, lambda x: x, "P-Weights", fig_titles)
