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
import json

from train import train
from evaluate import evaluate
from datasets import cifar, mnist
import numpy as np
import models.params
from models import pixeldp_cnn, pixeldp_resnet
import tensorflow as tf

import utils
from flags import FLAGS

def run_one():
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if FLAGS.dataset == 'mnist':
        image_size      = 28
        num_classes     = 10
        relu_leakiness  = 0.0
        lrn_rate        = 0.01
        lrn_rte_changes = []
        lrn_rte_vals    = []
        if FLAGS.mode == 'train':
            batch_size = 128
            n_draws    = 1
        elif FLAGS.mode == 'eval':
            batch_size = 100
            n_draws    = 500
    else:
        lrn_rate        = 0.1
        lrn_rte_changes = [40000, 60000, 80000]
        lrn_rte_vals    = [0.01, 0.001, 0.0001]
        if FLAGS.mode == 'train':
            batch_size = 128
            n_draws    = 1
        elif FLAGS.mode == 'eval':
            batch_size = 4
            n_draws    = 500

    if FLAGS.dataset == 'cifar10':
        image_size     = 32
        num_classes    = 10
        relu_leakiness = 0.1
    elif FLAGS.dataset == 'cifar100':
        image_size     = 32
        num_classes    = 100
        relu_leakiness = 0.1

    # noise_schemes for pixeldp_img_noise:
    #   - l1: L1 attack bound, L1 sensitivity (Laplace mech.).
    #   - l2: L2 attack bound, L2 sensitivity (Gaussian mech.).
    # noise_schemes for pixeldp_conv1:
    #   - l1_l1:    L1 attack bound with L1 sensitivity, reparametrization
    #               trick (Laplace).
    #   - l1_l1_s1: L1 attack bound with L1 sensitivity, sensitivity=1
    #               (Laplace).
    #   - l1_l2:    L1 attack bound with L2 sensitivity, reparametrization
    #               (Gaussian).
    #   - l1_l2_s1: L1 attack bound with L2 sensitivity, sensitivity=1
    #               (Gaussian).
    #   - l2_l2_s1: L2 attack bound with L2 sensitivity, sensitivity<=1
    #               (Gaussian).
    hps = models.params.HParams(batch_size=batch_size,
                        num_classes=num_classes,
                        image_size=image_size,
                        lrn_rate=lrn_rate,
                        lrn_rte_changes=lrn_rte_changes,
                        lrn_rte_vals=lrn_rte_vals,
                        num_residual_units=4,
                        use_bottleneck=False,
                        weight_decay_rate=0.0002,
                        relu_leakiness=relu_leakiness,
                        optimizer='mom',
                        image_standardization=False,
                        dropout=False,
                        n_draws=n_draws,
                        dp_epsilon=1.0,
                        dp_delta=0.05,
                        attack_norm_bound=0.3,
                        attack_norm='l2',
                        sensitivity_norm='l2',
                        sensitivity_control_scheme='bound',  # bound or optimize
                        noise_after_n_layers=1,
                        layer_sensitivity_bounds=['l2_l2'],
                        noise_after_activation=True,
                        )

    # _model can be: pixeldp_cnn or pixeldp_resnet
    _model = pixeldp_resnet
    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps, _model)
        elif FLAGS.mode == 'eval':
            evaluate(hps, _model)

def main(_):
    run_one()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
