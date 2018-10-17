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
import attacks
from attacks import params

def accuracy_under_attack(truth, adv_output_sums):
    true_n = 0
    tot_n  = 0
    for i, t in enumerate(truth):
        tot_n += 1
        correct_pred = True
        for output_scores in adv_output_sums[i]:
            output = np.argmax(output_scores)
            if output != t:
                correct_pred = False
                break
        if correct_pred:
            true_n += 1

    return true_n / tot_n

def accuracy(truth, output_sum):
    return plots.utils.accuracy(
            np.array(truth) == np.argmax(np.array(output_sum), axis=1))

def plot(file_name,
         models, params, attack_params,
         x_range=(0, 1), x_ticks=None, dir_name=None,
         expectation_layer='argmax', label_attack=False):

    if dir_name == None:
        dir_name = FLAGS.models_dir

    curves_y = []
    attack_y = []
    for model, param, model_attack_params in zip(models, params, attack_params):
        if x_ticks == None:
            x = []
        else:
            x = x_ticks

        curve_y = [0]

        model_dir = os.path.join(dir_name, models.params.name_from_params(model, param))
        for attack_param in model_attack_params:
            if attack_param.max_attack_size == 0:
                # This is not an attack...
                continue

            if x_ticks == None:
                x.append(attack_param.max_attack_size)

            attack_dir = os.path.join(model_dir, 'attack_results',
                    attacks.params.name_from_params(attack_param))

            with open(os.path.join(attack_dir, "eval_data.json")) as f:
                eval_data = json.loads(f.read())

            if 'baseline_acc' in eval_data:
                baseline_acc = eval_data['baseline_acc']
            else:
                truth  = eval_data['pred_truth']
                output_sum      = eval_data['{}_sum'.format(expectation_layer)]
                adv_output_sums = eval_data['adv_{}_sum'.format(expectation_layer)]
                baseline_acc = accuracy(truth, output_sum)

            curve_y[0] = baseline_acc

            if attack_param.attack_methodolody == 'carlini':
                if x_ticks == None:
                    raise ValueError("Carlini attack needs x_ticks.")
                ys = [baseline_acc]
                for t in x:
                    # The min is in case of multiple restarts
                    y = [min(l) > t for l in eval_data['adversarial_norm']]
                    curve_y.append(sum(y) / len(y))
                attack_label = "Carlini"
            else:
                acc = accuracy_under_attack(truth, adv_output_sums)
                curve_y.append(acc)
                attack_label = "PGD"

        attack_y.append(attack_label)
        curves_y.append(curve_y)

    x = [0] + x  # for the baseline


    plt.clf()
    fig, (ax) = plt.subplots(1, 1, figsize=plots.utils.figsize, tight_layout=True)
    x_offset = max(x_range)*0.01
    plt.xlim((x_range[0]-x_offset, x_range[1]+x_offset))
    plt.ylim((-0.02, 1.0))
    artists = []

    curves_labels = []
    for i, y in enumerate(curves_y):
        if label_attack:
            # The curve label should be the attack used and not the model.
            label = attack_y[i]
        else:
            label = plots.utils.label(models[i], params[i])
        curves_labels.append(label)
        art, = plt.plot(
            x, y,
            color=plots.utils.color(models[i], params[i]),
            linestyle=plots.utils.linestyle(models[i], params[i]),
            linewidth=plots.utils.line_thickness,
            label=label,
            marker=plots.utils.markerstyle(models[i], params[i]),
            markersize=plots.utils.markersize
        )
        artists.append([None, None, art])

    ax.set(xlabel=r'Empirical attack bound $L_{attack}$ (2-norm)', ylabel=r'Accuracy')
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
        f.write("{}\n".format(", ".join(curves_labels)))
        for i, _x in enumerate(x):
            robust_accs = []
            for curve in curves_y:
                robust_accs.append(curve[i])

            f.write("{:.3f}".format(_x))
            for racc in robust_accs:
                f.write(", {:.4f}".format(racc))
            f.write("\n")

