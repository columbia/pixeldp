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

from datasets import cifar, mnist
import numpy as np
import models.params
from models import pixeldp_cnn, pixeldp_resnet
import tensorflow as tf

import utils

from flags import FLAGS

def evaluate(hps, model, dir_name=None, rerun=False):
    """Evaluate the ResNet and log prediction counters to compute
    sensitivity."""
    if dir_name == None:
        dir_name = FLAGS.data_dir + "/" + FLAGS.model_dir

    if os.path.isfile(dir_name + "/eval_data.json") and not rerun:
        # run only new models
        return

    if FLAGS.dataset == 'mnist':
        images, labels = mnist.build_input(
            FLAGS.data_path,
            hps.batch_size,
            hps.image_standardization,
            FLAGS.mode
        )
    elif FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100':
        images, labels = cifar.build_input(
            FLAGS.dataset,
            FLAGS.data_path,
            hps.batch_size,
            hps.image_standardization,
            FLAGS.mode
        )
    model = model.Model(hps, images, labels, FLAGS.mode)
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(dir_name)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    try:
        ckpt_state = tf.train.get_checkpoint_state(dir_name)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    # Make predictions on the dataset, keep the label distribution
    data = {
        'predictions': [],
        'pred_truth':  [],
    }
    total_prediction, correct_prediction = 0, 0
    eval_data_size   = FLAGS.eval_data_size
    eval_batch_size  = hps.batch_size
    eval_batch_count = int(eval_data_size / eval_batch_size)
    for i in six.moves.range(eval_batch_count):
        if FLAGS.dataset == 'mnist':
            args = {model.noise_scale: 1.0}
        elif FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100':
            args = {model.noise_scale: 1.0}

        (
            summaries,
            loss,
            predictions,
            truth,
            train_step,
         ) = sess.run(
            [
                model.summaries,
                model.cost,
                model.predictions,
                model.labels,
                model.global_step,
            ],
            args)

        print("Done: {}/{}".format(eval_batch_size*i, eval_data_size))
        truth = np.argmax(truth, axis=1)[:hps.batch_size]
        prediction_votes = np.zeros([hps.batch_size, hps.num_classes])
        predictions = np.argmax(predictions, axis=1)
        for i in range(hps.n_draws):
            for j in range(hps.batch_size):
                prediction_votes[j, predictions[i*hps.batch_size + j]] += 1
        predictions = np.argmax(prediction_votes, axis=1)

        data['predictions'] += prediction_votes.tolist()
        data['pred_truth']  += (truth == predictions).tolist()

        print("{} / {}".format(np.sum(truth == predictions), len(predictions)))

        correct_prediction += np.sum(truth == predictions)
        total_prediction   += predictions.shape[0]

        current_precision = 1.0 * correct_prediction / total_prediction
        print(current_precision)
        print()

    # For Parseval, get true sensitivity, use to rescale the actual attack
    # bound as the nosie assumes this to be 1 but often it is not.
    if hps.noise_scheme == 'l2_l2_s1':
        # Parseval updates usually have a sensitivity higher than 1
        # despite the projection: we need to rescale when computing
        # sensitivity.
        sensitivity_multiplier = float(sess.run(
            model.sensitivity_multiplier,
            {model.noise_scale: 1.0}
        ))
    else:
        sensitivity_multiplier = 1.0
    with open(dir_name + "/sensitivity_multiplier.json", 'w') as f:
        d = [sensitivity_multiplier]
        f.write(json.dumps(d))

    # Compute robustness and add it to the eval data.
    dp_mechs = {
        'l2_l2_s1': 'gaussian',
        'l1_l2_s1': 'gaussian',
        'l1_l1_s1': 'laplace',
        'l1_l1':    'laplace',
        'l1_l2':    'gaussian',
        'l2':       'gaussian',
        'l1':       'laplace',
    }
    robustness = [utils.robustness_size(
        counts=x,
        dp_attack_size=hps.attack_norm_bound,
        dp_epsilon=1.0,
        dp_delta=0.05,
        dp_mechanism=dp_mechs[hps.noise_scheme]
        ) / sensitivity_multiplier for x in data['predictions']]
    data['robustness'] = robustness
    data['sensitivity_mult_used'] = sensitivity_multiplier

    # Log eval data
    with open(dir_name + "/eval_data.json", 'w') as f:
        f.write(json.dumps(data))

    # Print stuff
    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))
    summary_writer.flush()

