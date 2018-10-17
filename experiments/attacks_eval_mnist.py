"""Run attack comparison.
"""
import time, os, math
from multiprocessing import Pool

import models
import models.params
from datasets import cifar, mnist, svhn
from models.train import train
from models.evaluate import evaluate
from models import pixeldp_cnn, pixeldp_resnet
import attacks
from attacks import pgd, carlini, params, train_attack, evaluate_attack
import plots.plot_robust_accuracy
import plots.plot_accuracy_under_attack

from experiments import train_eval_model, train_eval_attack

import tensorflow as tf
import numpy as np

from flags import FLAGS

def run():
    param_dict = {
        'name_prefix': '',
        'steps_num': 40000,
        'eval_data_size': 10000,
        'image_size': 28,
        'n_channels': 1,
        'num_classes': 10,
        'relu_leakiness': 0.0,
        'lrn_rate': 0.1,
        'lrn_rte_changes': [30000],
        'lrn_rte_vals': [0.01],
        'num_residual_units': 4,
        'use_bottleneck': False,
        'weight_decay_rate': 0.0002,
        'optimizer': 'mom',
        'image_standardization': False,
        'dp_epsilon': 1.0,
        'dp_delta': 0.05,
        'robustness_confidence_proba': 0.05,
        'attack_norm': 'l2',
        'sensitivity_norm': 'l2',
        'sensitivity_control_scheme': 'bound',  # bound or optimize
        'layer_sensitivity_bounds': ['l2_l2'],
        'noise_after_activation': True,
        'parseval_loops': 10,
        'parseval_step': 0.0003,
    }

    parallelizable_arguments_list = []
    num_gpus = max(1, FLAGS.num_gpus)

    # First, create all params for train/eval models.
    for model_name in ["pixeldp_cnn"]:
        for attack_norm_bound in [0.1]:
            for noise_after_n_layers in [1]:
                if attack_norm_bound == 0.0 and noise_after_n_layers > -1:
                    continue  # The baseline can only have -1.
                if attack_norm_bound > 0.0 and noise_after_n_layers < 0:
                    continue  # PixelDP nets need a noise layer at position >= 0.

                param_dict['parseval_loops']       = math.ceil(100 * attack_norm_bound)
                param_dict['attack_norm_bound']    = attack_norm_bound
                param_dict['noise_after_n_layers'] = noise_after_n_layers
                parallelizable_arguments_list.append(
                    (
                        'mnist',
                        model_name,
                        dict(param_dict),
                        len(parallelizable_arguments_list) % num_gpus
                    )
                )


    # Run train/eval of models.
    print("\nTrain/Eval models:: Experiments: {}".\
          format(parallelizable_arguments_list))
    print("Train/Eval models:: Total experiments: {}".\
          format(len(parallelizable_arguments_list)))
    print("Train/Eval models:: Running on {} GPUs\n\n".format(num_gpus))
    results = []
    for i in range(0, len(parallelizable_arguments_list), num_gpus):
        p = Pool(processes=num_gpus)
        current = p.map(train_eval_model, parallelizable_arguments_list[i:min(i+num_gpus,len(parallelizable_arguments_list))])
        results.extend(current)
        p.close()
        p.join()
        time.sleep(5)

    # Second, create all params for train/eval attacks on models.
    parallelizable_arguments_list = []

    attack_param_dict = {
        'restarts': 15,
        'n_draws_attack': 50,
        'n_draws_eval':   500,
        'attack_norm': 'l2',
        'max_attack_size': -1,
        'num_examples': 1000,
        'attack_methodolody': 'pgd',
        'targeted': False,
        'sgd_iterations': 100,
        'use_softmax': False,
    }

    for (hps, model_name) in results:
        for attack_size in np.arange(0.1, 4.2, 0.25).tolist():
            attack_size = round(attack_size, 2)
            attack_param_dict['max_attack_size'] = attack_size
            parallelizable_arguments_list.append(
                (
                    'mnist',
                    hps,
                    model_name,
                    dict(attack_param_dict),
                    len(parallelizable_arguments_list) % num_gpus
                )
            )

        attack_param_dict['max_attack_size'] = 4.0
        attack_param_dict['restarts'] = 1
        attack_param_dict['attack_methodolody'] = "carlini"
        parallelizable_arguments_list.append(
            (
                'mnist',
                hps,
                model_name,
                dict(attack_param_dict),
                len(parallelizable_arguments_list) % num_gpus
            )
        )

    # Run train/eval of attracks on models.
    print("\nTrain/Eval attacks:: Experiments: {}".\
          format(parallelizable_arguments_list))
    print("Train/Eval attacks:: Total experiments: {}".\
          format(len(parallelizable_arguments_list)))
    print("Train/Eval attacks:: Running on {} GPUs\n\n".format(num_gpus))
    results = []
    for i in range(0, len(parallelizable_arguments_list), num_gpus):
        p = Pool(processes=num_gpus)
        current = p.map(train_eval_attack, parallelizable_arguments_list[i:min(i+num_gpus,len(parallelizable_arguments_list))])
        results.extend(current)
        p.close()
        p.join()
        print("Finished experiments: {}/{}".\
              format(len(results), len(parallelizable_arguments_list)))
        time.sleep(5)

    # Retrieve all results after evaluating the attacks and order them for
    # ploting scripts.
    _models = []
    _params = []
    _attack_params = []
    for (hps, attack_params, model_name) in results:
        if hps.attack_norm_bound not \
            in list(map(lambda x: x.attack_norm_bound, _params)):
            for _ in range(2):
                _models.append(models.module_from_name(model_name))
                _params.append(hps)
                _attack_params.append([])
        if attack_params.attack_methodolody == 'carlini':
            _attack_params[-1].append(attack_params)
        else:
            _attack_params[-2].append(attack_params)

    # Plot robust accuracy results
    dir_name = os.path.join(FLAGS.models_dir, 'mnist')
    # Plot accuracy under attack
    plots.plot_accuracy_under_attack.plot("attack_eval_argmax",
                                          _models,
                                          _params,
                                          _attack_params,
                                          x_range=(0, 4.1),
                                          x_ticks=[round(a, 2) for a in
                                              np.arange(0.1, 4.2, 0.25).tolist()],
                                          dir_name=dir_name,
                                          label_attack=True)
    #  plots.plot_accuracy_under_attack.plot("accuracy_under_attack_softmax",
                                          #  _models,
                                          #  _params,
                                          #  _attack_params,
                                          #  x_range=(0, 1.4),
                                          #  dir_name=dir_name,
                                          #  expectation_layer='softmax',
                                          #  label_attack=True)


def main(_):
    run()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


