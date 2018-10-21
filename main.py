# Copyright 2016 The Pixeldp Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Based on https://github.com/tensorflow/models/tree/master/research/resnet

"""ResNet Train/Eval module.
"""
import time
import six
import sys
import os
import json, math

from models import train
from models import evaluate
from datasets import cifar, mnist, svhn
import numpy as np
import models.params
from models import pixeldp_cnn, pixeldp_resnet, madry
import tensorflow as tf
import plots.plot_robust_accuracy
import plots.plot_accuracy_under_attack
import plots.plot_robust_precision_under_attack

import attacks
from attacks import train_attack, evaluate_attack, pgd, carlini, params, carlini_robust_precision, evaluate_attack_carlini_robust_prec

from flags import FLAGS

def run_one():
    # Manual runs support cpu or 1 gpu
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    else:
        dev = '/gpu:0'

    if FLAGS.dataset == 'mnist':
        _model = pixeldp_cnn

        steps_num       = 40000
        eval_data_size  = 10000
        image_size      = 28
        n_channels      = 1
        num_classes     = 10
        relu_leakiness  = 0.0
        lrn_rate        = 0.01
        lrn_rte_changes = [30000]
        lrn_rte_vals    = [0.01]
        if FLAGS.mode == 'train':
            batch_size = 128
            n_draws    = 1
        elif FLAGS.mode == 'eval':
            batch_size = 25
            n_draws    = 2000
    elif FLAGS.dataset == 'svhn':
        _model = pixeldp_resnet

        steps_num       = 60000
        eval_data_size  = 26032
        image_size      = 32
        n_channels      = 3
        num_classes     = 10
        relu_leakiness  = 0.0
        lrn_rate        = 0.01
        lrn_rte_changes = [20000, 40000, 50000]
        lrn_rte_vals    = [0.01, 0.001, 0.0001]
        if FLAGS.mode == 'train':
            batch_size = 128
            n_draws    = 1
        elif FLAGS.mode == 'eval':
            batch_size = 25
            n_draws    = 2000
    else:
        steps_num       = 90000
        eval_data_size  = 10000
        lrn_rate        = 0.1
        lrn_rte_changes = [40000, 60000, 80000]
        lrn_rte_vals    = [0.01, 0.001, 0.0001]
        if FLAGS.mode == 'train':
            batch_size = 128
            n_draws    = 1
        elif FLAGS.mode == 'eval':
            batch_size = 1
            n_draws    = 2000

    if FLAGS.dataset == 'cifar10':
        _model = pixeldp_resnet

        image_size     = 32
        n_channels      = 3
        num_classes    = 10
        relu_leakiness = 0.1
    elif FLAGS.dataset == 'cifar100':
        _model = pixeldp_resnet

        image_size     = 32
        n_channels      = 3
        num_classes    = 100
        relu_leakiness = 0.1

    if FLAGS.mode in ['attack', 'attack_eval', 'plot']:
        batch_size = 1
        n_draws    = 10

    compute_robustness = True

    # See doc in ./models/params.py
    L = 0.1
    hps = models.params.HParams(
            name_prefix="",
            batch_size=batch_size,
            num_classes=num_classes,
            image_size=image_size,
            n_channels=n_channels,
            lrn_rate=lrn_rate,
            lrn_rte_changes=lrn_rte_changes,
            lrn_rte_vals=lrn_rte_vals,
            num_residual_units=4,
            use_bottleneck=False,
            weight_decay_rate=0.0002,
            relu_leakiness=relu_leakiness,
            optimizer='mom',
            image_standardization=False,
            n_draws=n_draws,
            dp_epsilon=1.0,
            dp_delta=0.05,
            robustness_confidence_proba=0.05,
            attack_norm_bound=L,
            attack_norm='l2',
            sensitivity_norm='l2',
            sensitivity_control_scheme='bound',  # bound or optimize
            noise_after_n_layers=1,
            layer_sensitivity_bounds=['l2_l2'],
            noise_after_activation=True,
            parseval_loops=10,
            parseval_step=0.0003,
            steps_num=steps_num,
            eval_data_size=eval_data_size,
    )

    #  atk = pgd
    atk = carlini
    #  atk = carlini_robust_precision
    if atk == carlini_robust_precision:
        attack_params = attacks.params.AttackParamsPrec(
            restarts=1,
            n_draws_attack=20,
            n_draws_eval=500,
            attack_norm='l2',
            max_attack_size=5,
            num_examples=1000,
            attack_methodolody=attacks.name_from_module(atk),
            targeted=False,
            sgd_iterations=100,
            use_softmax=False,
            T=0.01
        )
    else:
        attack_params = attacks.params.AttackParams(
            restarts=1,
            n_draws_attack=20,
            n_draws_eval=500,
            attack_norm='l2',
            max_attack_size=5,
            num_examples=1000,
            attack_methodolody=attacks.name_from_module(atk),
            targeted=False,
            sgd_iterations=100,
            use_softmax=True
        )

    #  _model = pixeldp_cnn
    #  _model = pixeldp_resnet
    #  _model = madry

    if _model == madry:
        madry.Model.maybe_download_and_extract(FLAGS.models_dir)
        hps = models.params.update(hps, 'batch_size', 200)
        hps = models.params.update(hps, 'n_draws', 1)
        attack_params = attacks.params.update(attack_params, 'n_draws_attack', 1)
        attack_params = attacks.params.update(attack_params, 'n_draws_eval', 1)
        compute_robustness = False

    if FLAGS.mode == 'train':
        train.train(hps, _model, dev=dev)
    elif FLAGS.mode == 'eval':
        evaluate.evaluate(hps, _model, compute_robustness=compute_robustness,
                dev=dev)
    elif FLAGS.mode == 'attack':
        train_attack.train_one(
                FLAGS.dataset,
                _model,
                hps,
                atk,
                attack_params,
                dev=dev)

        tf.reset_default_graph()
    elif FLAGS.mode == 'attack_eval':
        if attack_params.attack_methodolody == 'carlini_robust_precision':
            evaluate_attack_carlini_robust_prec.evaluate_one(
                    FLAGS.dataset,
                    _model,
                    hps,
                    atk,
                    attack_params,
                    dev=dev)
        else:
            evaluate_attack.evaluate_one(
                    FLAGS.dataset,
                    _model,
                    hps,
                    atk,
                    attack_params,
                    dev=dev)
    elif FLAGS.mode == 'plot':
        ms   = []
        ps   = []
        atks = [[]]
        robust_ms   = [_model]
        robust_ps   = [hps]
        robust_atks = [[attack_params]]
        #  plots.plot_robust_accuracy.plot("test_robust_acc", None, None, ms, ps)
        plots.plot_accuracy_under_attack.plot("test_acc_under_atk",
                robust_ms, robust_ps, robust_atks, x_ticks=[x/10 for x in range(1,16)])
        #  plots.plot_robust_precision_under_attack.plot("test_robust_prec_under_atk",
                #  ms, ps, atks,
                #  robust_ms, robust_ps, robust_atks,
                #  x_range=(0, 2),
                #  x_ticks=[x/10 for x in range(1,21)])


def main(_):
    run_one()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
