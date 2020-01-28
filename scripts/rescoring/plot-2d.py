#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt


def parse_line(line):
    #  0    1     2 3   4 5     6  7    8   9    10  11  12 13
    #  %WER 10.35 [ 852 / 8234, 77 ins, 136 del, 639 sub ] /path/pick-1.0-17.0-20-0.5/wer_1_0.0

    fields = line.split()
    wer = fields[1]
    nb_errs = fields[3]
    nb_words = fields[5][:-1]
    nb_ins = fields[6]
    nb_dels = fields[8]
    nb_subs = fields[10]
    vals = {
        'wer': float(wer),
        'nb_errs': int(nb_errs),
        'nb_words': int(nb_words),
        'nb_ins': int(nb_ins),
        'nb_dels': int(nb_dels),
        'nb_subs': int(nb_subs),
    }

    fn = fields[13]

    parts_of_path = fn.split('/')
    assert parts_of_path[-1] == 'wer_1_0.0'

    coords_field = parts_of_path[-2]
    coords_fields = coords_field.split('-')
    assert coords_fields[0] == 'pick'

    coords = tuple(float(c) for c in coords_fields[1:])

    return coords, vals


if __name__ == '__main__':
    coord1 = 2
    coord2 = 3
    key = 'nb_errs'

    default_coords = (1.0, 17.0, 17.0, 0.0)

    measurements = {}
    for line in sys.stdin:
        coords, vals = parse_line(line)
        measurements[coords] = vals

    default_vals = measurements[default_coords]

    xs = []
    ys = []
    cs = []
    for k in measurements:
        xs.append(k[coord1])
        ys.append(k[coord2])
        cs.append(measurements[k][key] - default_vals[key])

    max_range = max(abs(x) for x in cs)
    print(max_range)

    plt.figure()
    # plt.scatter(xs, ys, c=cs, cmap='seismic', vmin=-max_range, vmax=max_range)
    plt.tricontourf(xs, ys, cs, 100, cmap='seismic', vmin=-max_range, vmax=max_range)
    plt.plot(xs, ys, 'ko ', markersize=1.0)
    plt.show()
