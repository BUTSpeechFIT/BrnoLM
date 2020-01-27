import numpy as np
import copy


def area_under_curve(xs_in, ys_in):
    assert(len(xs_in) == len(ys_in))

    xs = list(copy.deepcopy(xs_in))
    ys = list(copy.deepcopy(ys_in))

    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 1.0)

    if ys[-1] > 0.0:
        xs.append(1.0)
        ys.append(0.0)

    running_sum = 0.0

    for i in range(len(xs)-1):
        x_len = xs[i+1] - xs[i]
        avg_y = (ys[i] + ys[i+1])/2
        running_sum += x_len * avg_y

    return running_sum


def eer(xs, ys):
    assert(len(xs) == len(ys))

    eer = float('nan')
    for i in range(len(xs)-1):
        if xs[i] < ys[i] and xs[i+1] >= ys[i+1]:
            d_i = abs(xs[i] - ys[i])
            d_ip1 = xs[i+1] - ys[i+1]
            lambda_i = d_i/(d_i + d_ip1)
            eer = (1.0 - lambda_i) * xs[i] + lambda_i * xs[i+1]

    return eer


def det_points_from_score_tg(score_tg):
    nb_trials = len(score_tg)
    nb_same = sum(s[1] for s in score_tg)
    nb_different = nb_trials - nb_same

    sorted_score_tg = sorted(score_tg, key=lambda s: s[0], reverse=True)

    nb_correct_same = 0
    nb_correct_different = nb_different
    nb_false_alarms = 0
    nb_misses = nb_same
    mis_fas = [[nb_misses/nb_trials, nb_false_alarms/nb_trials]]

    for s in sorted_score_tg:
        if s[1] == 1:
            nb_misses -= 1
            nb_correct_same += 1
        else:
            nb_false_alarms += 1
            nb_correct_different -= 1

        mis_fas.append([nb_misses/nb_trials, nb_false_alarms/nb_trials])

    return mis_fas, [s[0] for s in sorted_score_tg]


def subsampling_indices(length, max_points):
    ''' Ensures that both the first and the last element are included.
    '''
    all_indices = list(range(length))

    subsampling_coeff_exact = (len(all_indices) - 1) / (max_points-1)
    if subsampling_coeff_exact > 1.0 and subsampling_coeff_exact < 2.0:
        inverse_subsampling_coeff_exact = (len(all_indices) - 2) / (len(all_indices) - max_points)
        inverse_subsampling_coeff = int(inverse_subsampling_coeff_exact)
        to_drop = list(range(1, len(all_indices)-1, inverse_subsampling_coeff))
        return [x for i, x in enumerate(all_indices) if i not in to_drop]
    else:
        subsampling_coeff = int(subsampling_coeff_exact)
        return all_indices[0:-1:subsampling_coeff] + [all_indices[-1]]


def pick(the_list, indices):
    return [the_list[i] for i in indices]


def subsample_list(the_list, max_points):
    ''' Ensures that both the first and the last element are included.
    '''
    indices = subsampling_indices(len(the_list), max_points)
    return pick(the_list, indices)


class DETCurve:
    def __init__(self, score_tg, baseline, max_det_points=0):
        self._baseline = baseline

        nb_trials = len(score_tg)
        nb_same = sum(s[1] for s in score_tg)
        nb_different = nb_trials - nb_same

        self._max_miss_rate = nb_same / nb_trials
        self._max_fa_rate = nb_different / nb_trials

        print("# positive trials: {} ({:.1f} %)".format(nb_same, 100.0*nb_same/nb_trials))
        print("# negative trials: {} ({:.1f} %)".format(nb_different, 100.0*nb_different/nb_trials))

        mis_fas, sorted_scores = det_points_from_score_tg(score_tg)

        if max_det_points > 0 and max_det_points < len(mis_fas):
            self._nb_clusterings = subsampling_indices(len(mis_fas), max_det_points)
            self._scores = pick(sorted_scores, self._nb_clusterings[:-1] + [len(sorted_scores)-1])
            mis_fas = pick(mis_fas, self._nb_clusterings)
        else:
            self._nb_clusterings = range(len(mis_fas))
            self._scores = sorted_scores

        self._miss_rate = [msfa[0] for msfa in mis_fas]
        self._fa_rate = [msfa[1] for msfa in mis_fas]

    def textual_report(self):
        report = ""
        area_line_fmt = "Area under DET curve (in linspace): {:.5f}"
        eer_line_fmt = "EER: {:.2f} %"

        system_au_det = area_under_curve(self._fa_rate, self._miss_rate)
        system_eer = eer(self._fa_rate, self._miss_rate)

        if self._baseline:
            area_line_fmt += " / {:.5f} / {:.2f} %"
            eer_line_fmt += " / {:.2f} % / {:.2f} %"

            baseline_au_det = self._max_miss_rate * self._max_fa_rate / 2.0
            baseline_eer = self._max_miss_rate * self._max_fa_rate / (self._max_miss_rate + self._max_fa_rate)

            report += area_line_fmt.format(
                system_au_det,
                baseline_au_det,
                100.0 * (1.0 - system_au_det/baseline_au_det)
            ) + '\n'
            report += eer_line_fmt.format(
                100.0*system_eer,
                100.0*baseline_eer,
                100.0 * (1.0 - system_eer/baseline_eer)
            ) + '\n'

        else:
            report += area_line_fmt.format(system_au_det) + '\n'
            report += eer_line_fmt.format(100.0*system_eer) + '\n'

        return report

    def plot(self, log_axis, scaled_axis, eer_line, filename):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if log_axis:
            plt_func = plt.loglog
        else:
            plt_func = plt.plot
        plt_func(self._miss_rate, self._fa_rate, label='System')

        for i, score, xy in zip(self._nb_clusterings, self._scores, zip(self._miss_rate, self._fa_rate)):
            label = '{} ({:.2f})'.format(i, score)
            ax.annotate((label), xy=xy, textcoords='data', fontsize='small')

        if self._baseline:
            xs = np.linspace(0, self._max_miss_rate)
            ys = np.linspace(self._max_fa_rate, 0)
            plt_func(xs, ys, label='Baseline')

            plt.legend()

        if eer_line:
            endpoint = min([self._max_fa_rate, self._max_miss_rate])
            plt_func([0.0, endpoint], [0.0, endpoint], color='k', linestyle='-.', linewidth=0.75)

        if scaled_axis:
            plt.axis('scaled')
        plt.xlim(left=0.0)
        plt.ylim(bottom=0.0)
        plt.xlabel('miss rate')
        plt.ylabel('FA rate')

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
