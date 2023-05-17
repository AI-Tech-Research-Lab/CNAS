import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import cm
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 'large'


def plot_scores(results: dict, branches=5):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_ylim(-0.1, 1.05)
    ax2.set_ylim(-0.1, 1.05)

    ax1.set_ylabel('Branches scores')
    ax2.set_ylabel('Branches counters')
    ax2.set_xlabel('Threshold')

    keys = sorted(map(float, results.keys()))
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    legend = set()

    for i, k in enumerate(keys):
        k = str(k)
        r = results[k]

        scores = r['scores']
        counters = r['counters']
        tot = sum(counters.values())

        x = i - 0.4 + (0.8 / 6) * 6
        ax1.bar(x, scores['global'], width=0.8 / 6,
                color=colors[-1], label='Acc' if i == 0 else '')

        for j in range(5):
            kj = str(j)

            if kj in scores:
                x = i - 0.4 + (0.8 / 6) * j
                ax1.bar(x, scores[kj], width=0.8 / 6,
                        color=colors[j],
                        label='B {}'.format(j) if j not in legend else '')
                legend.add(j)

            if kj in counters:
                x = i - 0.4 + (0.8 / 5) * j
                ax2.bar(x, counters[kj] / tot, width=0.8 / 5,
                        color=colors[
                            j],
                        label='B {}'.format(j) if j not in legend else '')
                legend.add(j)

    for i in np.arange(0, 1.1, 0.1):
        ax2.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)
        ax1.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)

    ax2.set_xticklabels([''] + keys)
    fig.legend()

    return fig


def plot_heatmap(results: dict, branches=5):
    keys = sorted(map(float, results.keys()))
    # colors = ['b', 'g', 'r', 'c', 'm', 'k']
    # legend = set()

    matrix_scores = np.zeros((len(keys), branches + 1))
    matrix_counters = np.zeros((len(keys), branches))

    for i, k in enumerate(keys):
        k = str(k)
        r = results[k]

        scores = r['scores']
        counters = r['counters']
        tot = sum(counters.values())

        for j in range(branches):
            kj = str(j)

            matrix_scores[i][j] = scores.get(kj, 0)
            matrix_counters[i][j] = counters.get(kj, 0)

        matrix_scores[i][-1] = scores['global']

    tot = matrix_counters.sum(1)[0]

    matrix_counters /= tot

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    matrix_scores = np.around(matrix_scores * 100, 1)
    matrix_counters = np.around(matrix_counters * 100, 1)

    ax1.imshow(matrix_scores, cmap=cm.inferno_r, vmin=0, vmax=100)
    ax2.imshow(matrix_counters, cmap=cm.Blues, vmin=0, vmax=100)

    ax1.set_xlabel('Branch index')
    ax1.set_ylabel('Threshold')
    ax1.set_yticklabels([''] + keys)
    ax1.set_xticks(list(range(branches + 1)))
    ax1.set_xticklabels(list(range(1, branches + 1)) + ['Global'])

    ax2.set_xlabel('Branch index')
    ax2.set_ylabel('Threshold')
    ax2.set_yticklabels([''] + keys)

    ax2.set_xticks(list(range(branches)))
    ax2.set_xticklabels(list(range(1, branches + 1)))

    for i in range(len(keys)):
        for j in range(matrix_scores.shape[1]):
            if matrix_scores[i, j] != 0:
                text = ax1.text(j, i, matrix_scores[i, j],
                                ha="center", va="center", color="w"
                    if matrix_scores[i, j] >= 50 else 'k')

    for i in range(len(keys)):
        for j in range(matrix_counters.shape[1]):
            if matrix_counters[i, j] != 0:
                text = ax2.text(j, i, matrix_counters[i, j],
                                ha="center", va="center", color="w"
                    if matrix_counters[i, j] >= 50 else 'k')

    return fig1, fig2


def plot_lines(results: dict, costs: dict, branches_scores: dict, branches=5):
    keys = sorted(map(float, results.keys()))
    # colors = ['b', 'g', 'r', 'c', 'm', 'k']
    # legend = set()

    matrix_scores = np.zeros((len(keys), branches + 1))
    matrix_counters = np.zeros((len(keys), branches))

    for i, k in enumerate(keys):
        k = str(k)
        r = results[k]

        scores = r['scores']
        counters = r['counters']
        tot = sum(counters.values())

        for j in range(branches):
            kj = str(j)

            matrix_scores[i][j] = scores.get(kj, 0)
            matrix_counters[i][j] = counters.get(kj, 0)

        matrix_scores[i][-1] = scores['global']

    tot = matrix_counters.sum(1)[0]

    matrix_counters /= tot

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    matrix_scores = np.around(matrix_scores * 100, 1)
    matrix_counters = np.around(matrix_counters * 100, 1)

    for i in range(matrix_scores.shape[1]):
        p = matrix_scores[:, i]
        last = i == matrix_scores.shape[1] - 1
        if last:
            continue

        lw = mpl.rcParams['lines.linewidth'] if not last else \
            mpl.rcParams['lines.linewidth'] * 1.5

        label = 'Branch {}'.format(i + 1) if not last \
            else 'Global Accuracy'

        poly = np.polyfit(range(len(p)), p, 2)
        poly_y = np.poly1d(poly)(range(len(p)))
        poly_y = np.minimum(poly_y, 100)
        poly_y = np.maximum(poly_y, 0)

        ax1.plot(poly_y, label=label, linewidth=lw)

    ax1.axhline(branches_scores['final'] * 100,
                color='k', linestyle='--', label='Model final accuracy')

    for i in range(matrix_counters.shape[1]):
        p = matrix_counters[:, i]
        label = 'Branch {}'.format(i + 1)

        poly = np.polyfit(range(len(p)), p, 2)
        poly_y = np.poly1d(poly)(range(len(p)))
        poly_y = np.minimum(poly_y, 100)
        poly_y = np.maximum(poly_y, 0)

        ax2.plot(poly_y, label=label)

    total_cost = costs[str(branches - 1)]
    normalize_costs = {k: v / total_cost for k, v in costs.items()}
    normalize_costs_array = np.asarray(list(normalize_costs.values()))

    mx = 1
    mn = min(normalize_costs_array)

    gains = []
    # print(normalize_costs_array)

    for i, k in enumerate(keys):
        counters = matrix_counters[i]
        counters *= (1 - normalize_costs_array)
        # counters -= mn
        # counters /= (mx - mn)
        gains.append(sum(counters))
        # print(i, counters)

    # gains = 100 - gains
    # print(gains)

    poly = np.polyfit(range(len(gains)), gains, 2)
    poly_y = np.poly1d(poly)(range(len(gains)))
    poly_y = np.minimum(poly_y, 100)
    poly_y = np.maximum(poly_y, 0)

    ax3.plot(poly_y, color='b', label='Gains')
    ax3.axhline(branches_scores['final'] * 100,
                color='k', linestyle='--', label='Model final accuracy')

    poly = np.polyfit(range(len(gains)), matrix_scores[:, -1], 2)
    poly_y = np.poly1d(poly)(range(len(gains)))
    poly_y = np.minimum(poly_y, 100)
    poly_y = np.maximum(poly_y, 0)
    ax3.plot(poly_y, color='r', label='Accuracy')

    ax1.set_ylim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax3.set_ylim(-5, 105)

    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels(keys)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys)
    ax3.set_xticks(range(len(keys)))
    ax3.set_xticklabels(keys)

    # ax1.set_xticklabels(keys)
    # ax2.set_xticklabels(keys)

    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    ax3.grid(alpha=0.5)

    ax1.legend()
    ax3.legend()
    # ax2.legend()

    ax1.set_ylabel('Branch accuracy (%)')
    ax1.set_xlabel('Threshold')

    ax2.set_ylabel('Branch counters (%)')
    ax2.set_xlabel('Threshold')

    ax3.set_xlabel('Threshold')

    return fig1, fig2, fig3


def plot_statistics(results: dict, branches=5):
    correct = results['correct']
    incorrect = results['incorrect']

    fis = []

    keys = sorted(correct.keys())

    for key in keys:
        # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig, ax1 = plt.subplots(1, sharex=True)
        # ax1.set_ylim(-0.1, 1.05)
        # ax2.set_ylim(-0.1, 1.05)

        cor = np.asarray(correct[key])[:, None]
        inc = np.asarray(incorrect[key])[:, None]

        bottom, _, _ = ax1.hist(cor, bins=50, density=False, color='r',
                                alpha=.8, fill=False, histtype='step')
        ax1.hist(inc, bins=50, bottom=bottom, alpha=.8,
                 density=False, color='b', histtype='step')

        #ax2.hist(inc, bins=50, bottom=None, density=False, color='b', alpha=.5)

        ax1.set_xlabel('Confidence')

        fis.append(fig)
        # plt.show()
        # plt.close('all')

    return fis
    # for i, k in enumerate(keys):
    #     k = str(k)
    #     r = results[k]
    #
    #     scores = r['scores']
    #     counters = r['counters']
    #     tot = sum(counters.values())
    #
    #     x = i - 0.4 + (0.8 / 6) * 6
    #     ax1.bar(x, scores['global'], width=0.8 / 6,
    #             color=colors[-1], label='Acc' if i == 0 else '')
    #
    #     for j in range(5):
    #         kj = str(j)
    #
    #         if kj in scores:
    #             x = i - 0.4 + (0.8 / 6) * j
    #             ax1.bar(x, scores[kj], width=0.8 / 6,
    #                     color=colors[j],
    #                     label='B {}'.format(j) if j not in legend else '')
    #             legend.add(j)
    #
    #         if kj in counters:
    #             x = i - 0.4 + (0.8 / 5) * j
    #             ax2.bar(x, counters[kj] / tot, width=0.8 / 5,
    #                     color=colors[
    #                         j],
    #                     label='B {}'.format(j) if j not in legend else '')
    #             legend.add(j)
    #
    # for i in np.arange(0, 1.1, 0.1):
    #     ax2.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)
    #     ax1.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)
    #
    # ax2.set_xticklabels([''] + keys)
    # fig.legend()
    #
    # return fig


def my_app() -> None:
    paths = sys.argv[1:]

    for path in paths:

        for experiment_path in [f for f in os.listdir(path)
                                if re.match(r'exp_[0-9]', f)]:

            full_path = os.path.join(path, experiment_path)

            if os.path.exists(os.path.join(full_path, 'results.json')):
                # log.info('Model loaded')
                with open(os.path.join(full_path, 'results.json'), 'r') \
                        as json_file:
                    results = json.load(json_file)

                branches = len(results['branch_scores'])

                plot_dirs = os.path.join(full_path, 'plots')
                os.makedirs(plot_dirs, exist_ok=True)

                if 'binary_statistics' in results:
                    figs = plot_statistics(results['binary_statistics'],
                                           branches=branches)

                    for i, f in enumerate(figs):
                        f.savefig(os.path.join(plot_dirs,
                                               'stats{}.pdf'.format(i)),
                                  bbox_inches=None)

                        plt.close('all')

                if 'binary_statistics_threshold' in results:
                    r = results['binary_statistics_threshold']
                    for th in r:
                        figs = plot_statistics(r[th],
                                               branches=branches)

                        for i, f in enumerate(figs):
                            f.savefig(os.path.join(plot_dirs,
                                                   'stats{}_th{}.pdf'.
                                                   format(i, th)),
                                      bbox_inches=None)

                        plt.close('all')

                if 'binary_results' in results:
                    # binary_scores_fig = plot_scores(results['binary_results'],
                    #                                 branches=branches)
                    # binary_scores_fig.savefig(os.path.join(
                    #     plot_dirs, 'binary_results.pdf'),
                    #     bbox_inches=None)

                    f1, f2, f3 = plot_lines(results['binary_results'],
                                            results['costs'],
                                            results['branch_scores'],
                                            branches=branches)

                    f1.savefig(os.path.join(plot_dirs,
                                            'binary_scores.pdf'),
                               bbox_inches=None)
                    f2.savefig(os.path.join(plot_dirs,
                                            'binary_counters.pdf'),
                               bbox_inches=None)

                    f3.savefig(os.path.join(plot_dirs,
                                            'binary_gains.pdf'),
                               bbox_inches=None)

                    f1, f2 = plot_heatmap(results['binary_results'],
                                          branches=branches)
                    f1.savefig(os.path.join(plot_dirs,
                                            'binary_scores_hm.pdf'),
                               bbox_inches=None)
                    f2.savefig(os.path.join(plot_dirs,
                                            'binary_counters_hm.pdf'),
                               bbox_inches=None)

                    plt.close('all')

                if 'cumulative_results' in results:
                    # cumulative_scores_fig = plot_scores(
                    #     results['cumulative_results'],
                    #     branches=branches)
                    # cumulative_scores_fig.savefig(os.path.join(
                    #     plot_dirs, 'cumulative_results.pdf'),
                    #     bbox_inches=None)

                    f1, f2, f3 = plot_lines(results['cumulative_results'],
                                            results['costs'],
                                            results['branch_scores'],
                                            branches=branches)

                    f1.savefig(os.path.join(plot_dirs,
                                            'cumulative_scores.pdf'),
                               bbox_inches=None)

                    f2.savefig(os.path.join(plot_dirs,
                                            'cumulative_counters.pdf'),
                               bbox_inches=None)

                    f3.savefig(os.path.join(plot_dirs,
                                            'cumulative_gains.pdf'),
                               bbox_inches=None)

                    f1, f2 = plot_heatmap(results['cumulative_results'],
                                          branches=branches)
                    f1.savefig(os.path.join(plot_dirs,
                                            'cumulative_scores_hm.pdf'),
                               bbox_inches=None)
                    f2.savefig(os.path.join(plot_dirs,
                                            'cumulative_counters_hm.pdf'),
                               bbox_inches=None)
                    plt.close('all')

                if 'entropy_results' in results:

                    f1, f2, f3 = plot_lines(results['entropy_results'],
                                            results['costs'],
                                            results['branch_scores'],
                                            branches=branches)

                    f1.savefig(os.path.join(plot_dirs,
                                            'entropy_scores.pdf'),
                               bbox_inches=None)
                    f2.savefig(os.path.join(plot_dirs,
                                            'entropy_counters.pdf'),
                               bbox_inches=None)
                    f3.savefig(os.path.join(plot_dirs,
                                            'entropy_gains.pdf'),
                               bbox_inches=None)

                    f1, f2 = plot_heatmap(results['entropy_results'],
                                          branches=branches)
                    f1.savefig(os.path.join(plot_dirs,
                                            'entropy_scores_hm.pdf'),
                               bbox_inches=None)
                    f2.savefig(os.path.join(plot_dirs,
                                            'entropy_counters_hm.pdf'),
                               bbox_inches=None)
                    plt.close('all')


if __name__ == "__main__":
    my_app()
