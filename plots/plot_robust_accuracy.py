import scipy.stats
import numpy as np
import os, sys,  json, math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use(['classic', 'seaborn-deep', 'seaborn-dark'])
sns.set(style='whitegrid')

import models.params
from flags import FLAGS

import plots.utils

def plot(file_name,
         baseline_model, baseline_params,
         models, params,
         x_range=(0, 1, 0.1), dir_name=None,
         expectation_layer='argmax'):
    if dir_name == None:
        dir_name = FLAGS.models_dir

    x = np.arange(x_range[0], x_range[1]+x_range[2], x_range[2]).tolist()
    curves_y = []
    if baseline_model != None:
        model_dir = os.path.join(
                dir_name, models.params.name_from_params(baseline_model, baseline_params))
        with open(os.path.join(model_dir, "eval_data.json")) as f:
            eval_data = json.loads(f.read())

        pred_truth = eval_data['pred_truth_{}'.format(expectation_layer)]
        baseline_acc = plots.utils.accuracy(pred_truth)

    for model, param in zip(models, params):
        model_dir = os.path.join(dir_name, models.params.name_from_params(model, param))
        with open(os.path.join(model_dir, "eval_data.json")) as f:
            eval_data = json.loads(f.read())

        robustness = eval_data['robustness_from_{}'.format(expectation_layer)]
        pred_truth = eval_data['pred_truth_{}'.format(expectation_layer)]
        ps = plots.utils.robust_accuracy_survival_ps(x, pred_truth, robustness)

        curves_y.append(ps)

    plt.clf()
    fig, (ax) = plt.subplots(1, 1, figsize=plots.utils.figsize, tight_layout=True)
    x_offset = max(x_range)*0.01
    plt.xlim((x_range[0]-x_offset, x_range[1]+x_offset))
    plt.ylim((-0.02, 1.0))
    artists = []

    curves_labels = []
    for i, y in enumerate(curves_y):
        label = plots.utils.label(models[i], params[i])
        curves_labels.append(label)
        j = len(x) - 1
        while y[j-1] == 0:
            j -= 1
        art, = plt.plot(
                x[:j+1], y[:j+1],
            color=plots.utils.color(models[i], params[i]),
            linestyle=plots.utils.linestyle(models[i], params[i]),
            linewidth=plots.utils.line_thickness,
            label=label,
            marker=plots.utils.markerstyle(models[i], params[i]),
            markersize=plots.utils.markersize
        )
        artists.append([None, None, art])

    if baseline_model != None:
        #  artists = sorted(artists)
        art, = plt.plot(
            [0.0],
            [baseline_acc],
            label="Baseline",
            linestyle="none",
            color="0.0",
            clip_on=False,
            zorder=11,
            marker="X",
            markersize=plots.utils.markersize
        )
        artists = [[None, None, art]] + artists

    ax.set(xlabel=r'Robustness threshold $T$ (2-norm)', ylabel=r'Certified accuracy')
    l = plt.legend(
        handles=[x[2] for x in artists],
        loc=1,
        fontsize=plots.utils.legend_font_size
    )
    l.set_zorder(20)  # put the legend on top
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(plots.utils.ticks_font_size)
    ax.xaxis.label.set_fontsize(plots.utils.labels_font_size)
    ax.yaxis.label.set_fontsize(plots.utils.labels_font_size)

    plt.savefig(os.path.join(dir_name, "{}.pdf".format(file_name)),
            bbox_inches='tight', pad_inches=0)

    # Dump the data in a file to also have numbers
    with open(os.path.join(dir_name, "{}.txt".format(file_name)), 'w') as f:
        f.write("Baseline: {}\n".format(baseline_acc))
        f.write("{}\n".format(", ".join(curves_labels)))
        for i, _x in enumerate(x):
            robust_accs = []
            for curve in curves_y:
                robust_accs.append(curve[i])

            f.write("{:.3f}".format(_x))
            for racc in robust_accs:
                f.write(", {:.4f}".format(racc))
            f.write("\n")

