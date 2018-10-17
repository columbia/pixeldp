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

import datasets
import numpy as np
import models.params
from models import pixeldp_cnn, pixeldp_resnet
import tensorflow as tf

from models.utils import robustness

from flags import FLAGS

def evaluate(hps, model, dataset=None, dir_name=None, rerun=False,
             compute_robustness=True, dev='/cpu:0'):
    """Evaluate the ResNet and log prediction counters to compute
    sensitivity."""

    # Trick to start from arbitrary GPU  number
    gpu = int(dev.split(":")[1]) + FLAGS.min_gpu_number
    if gpu >= 16:
        gpu -= 16
    dev = "{}:{}".format(dev.split(":")[0], gpu)

    print("Evaluating model on dev:{}".format(dev))
    with tf.device(dev):
        if dir_name == None:
            dir_name = FLAGS.models_dir

        dir_name = os.path.join(dir_name, models.params.name_from_params(model, hps))

        if os.path.isfile(dir_name + "/eval_data.json") and not rerun:
            print("Skip eval of:{}".format(dir_name))
            # run only new models
            return

        if dataset == None:
            dataset = FLAGS.dataset

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = str(dev.split(":")[-1])
        sess = tf.Session(config=config)

        # Special treatment of imagenet: load inception + autoencoder
        if 'imagenet' in dir_name and hps.attack_norm_bound > .0:
            images, labels = datasets.build_input(
                dataset,
                FLAGS.data_path,
                hps.batch_size,
                hps.image_standardization,
                'eval'
            )
            autoencoder_dir_name = os.path.join(dir_name, "autoencoder_l2_l2_s1_{}_32_32_64_10_8_5_srd1221_srd1221_srd1221".format(hps.attack_norm_bound))
            autoencoder_params = json.load(
                open(os.path.join(autoencoder_dir_name, "params.json"), "r")
            )
            autoencoder_params['n_draws'] = hps.n_draws
            # hyperparams for autoencoder
            autoencoder_hps = tf.contrib.training.HParams()
            for k in autoencoder_params:
                autoencoder_hps.add_hparam(k, autoencoder_params[k])
            autoencoder_hps.batch_size = hps.batch_size*hps.n_draws
            autoencoder_hps.autoencoder_dir_name = autoencoder_dir_name
            from models import autoencoder_model
            autoencoder_model = autoencoder_model.Autoencoder(autoencoder_hps,
                                                              images,
                                                              images,
                                                              "eval")
            autoencoder_model.build_graph()
            autoencoder_variables = []
            for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                autoencoder_variables.append(k)
            autoencoder_saver = tf.train.Saver(autoencoder_variables)
            autoencoder_summary_writer = tf.summary.FileWriter(autoencoder_dir_name)
            try:
                autoencoder_ckpt_state = tf.train.get_checkpoint_state(autoencoder_dir_name)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
            print('Autoencoder: Loading checkpoint',
                            autoencoder_ckpt_state.model_checkpoint_path)
            autoencoder_saver.restore(sess,
                                      autoencoder_ckpt_state.model_checkpoint_path)
            # imagenet dataset loader returns images in [0, 1]
            images = 2*(autoencoder_model.output - 0.5)
        else:
            images, labels = datasets.build_input(
                dataset,
                FLAGS.data_path,
                hps.batch_size,
                hps.image_standardization,
                'eval'
            )

        tf.train.start_queue_runners(sess)
        model = model.Model(hps, images, labels, 'eval')
        model.build_graph()

        if hps.image_size == 299 and 'imagenet' in dir_name\
                and hps.attack_norm_bound > .0:
            inception_variables = []
            for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if k in autoencoder_variables and k.name != "global_step":
                    continue
                if k.name.startswith("DW-encoder") or k.name.startswith("b-encoder")\
                     or k.name.startswith("b-decoder"):
                    continue
                inception_variables.append(k)

            saver = tf.train.Saver(inception_variables)
        else:
            saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter(dir_name)

        try:
            ckpt_state = tf.train.get_checkpoint_state(dir_name)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        # Make predictions on the dataset, keep the label distribution
        data = {
            'argmax_sum': [],
            'softmax_sum': [],
            'softmax_sqr_sum': [],
            'pred_truth_argmax':  [],
            'pred_truth_softmax':  [],
        }
        total_prediction, correct_prediction_argmax, correct_prediction_logits = 0, 0, 0
        eval_data_size   = hps.eval_data_size
        eval_batch_size  = hps.batch_size
        eval_batch_count = int(eval_data_size / eval_batch_size)
        for i in six.moves.range(eval_batch_count):
            if model.noise_scale == None:
                args = {}  # For Madry and inception
            else:
                args = {model.noise_scale: 1.0}
            if 'imagenet' in dir_name and hps.attack_norm_bound > .0:
                args = {autoencoder_model.noise_scale: 1.0}
            (
                loss,
                softmax_predictions,
                truth
             ) = sess.run(
                [
                    model.cost,
                    model.predictions,
                    model.labels,
                ],
                args)
            print("Done: {}/{}".format(eval_batch_size*i, eval_data_size))
            truth = np.argmax(truth, axis=1)[:hps.batch_size]
            prediction_votes = np.zeros([hps.batch_size, hps.num_classes])
            softmax_sum = np.zeros([hps.batch_size, hps.num_classes])
            softmax_sqr_sum = np.zeros([hps.batch_size, hps.num_classes])

            predictions = np.argmax(softmax_predictions, axis=1)
            for i in range(hps.n_draws):
                for j in range(hps.batch_size):
                    prediction_votes[j, predictions[i*hps.batch_size + j]] += 1
                    softmax_sum[j] += softmax_predictions[i*hps.batch_size + j]
                    softmax_sqr_sum[j] += np.square(
                            softmax_predictions[i*hps.batch_size + j])

            predictions = np.argmax(prediction_votes, axis=1)
            predictions_logits = np.argmax(softmax_sum, axis=1)

            data['argmax_sum'] += prediction_votes.tolist()
            data['softmax_sum'] += softmax_sum.tolist()
            data['softmax_sqr_sum'] += softmax_sqr_sum.tolist()
            data['pred_truth_argmax']  += (truth == predictions).tolist()
            data['pred_truth_softmax'] += (truth == predictions_logits).tolist()

            print("From argamx: {} / {}".format(np.sum(truth == predictions), len(predictions)))
            print("From logits: {} / {}".format(np.sum(truth == predictions_logits), len(predictions)))

            correct_prediction_argmax += np.sum(truth == predictions)
            correct_prediction_logits += np.sum(truth == predictions_logits)
            total_prediction   += predictions.shape[0]

            current_precision_argmax = 1.0 * correct_prediction_argmax / total_prediction
            current_precision_logits = 1.0 * correct_prediction_logits / total_prediction
            print("Current precision from argmax: {}".format(current_precision_argmax))
            print("Current precision from logits: {}".format(current_precision_logits))
            print()

        # For Parseval, get true sensitivity, use to rescale the actual attack
        # bound as the nosie assumes this to be 1 but often it is not.
        # Parseval updates usually have a sensitivity higher than 1
        # despite the projection: we need to rescale when computing
        # sensitivity.
        if model.pre_noise_sensitivity() == None:
            sensitivity_multiplier = None
        else:
            sensitivity_multiplier = float(sess.run(
                model.pre_noise_sensitivity(),
                {model.noise_scale: 1.0}
            ))
        with open(dir_name + "/sensitivity_multiplier.json", 'w') as f:
            d = [sensitivity_multiplier]
            f.write(json.dumps(d))

        # Compute robustness and add it to the eval data.
        if compute_robustness:  # This is used mostly to avoid errors on non pixeldp DNNs
            dp_mechs = {
                'l2': 'gaussian',
                'l1': 'laplace',
            }
            robustness_from_argmax = [robustness.robustness_size_argmax(
                counts=x,
                eta=hps.robustness_confidence_proba,
                dp_attack_size=hps.attack_norm_bound,
                dp_epsilon=hps.dp_epsilon,
                dp_delta=hps.dp_delta,
                dp_mechanism=dp_mechs[hps.sensitivity_norm]
                ) / sensitivity_multiplier for x in data['argmax_sum']]
            data['robustness_from_argmax'] = robustness_from_argmax
            robustness_from_softmax = [robustness.robustness_size_softmax(
                tot_sum=data['softmax_sum'][i],
                sqr_sum=data['softmax_sqr_sum'][i],
                counts=data['argmax_sum'][i],
                eta=hps.robustness_confidence_proba,
                dp_attack_size=hps.attack_norm_bound,
                dp_epsilon=hps.dp_epsilon,
                dp_delta=hps.dp_delta,
                dp_mechanism=dp_mechs[hps.sensitivity_norm]
                ) / sensitivity_multiplier for i in range(len(data['argmax_sum']))]
            data['robustness_from_softmax'] = robustness_from_softmax

        data['sensitivity_mult_used'] = sensitivity_multiplier

        # Log eval data
        with open(dir_name + "/eval_data.json", 'w') as f:
            f.write(json.dumps(data))

        # Print stuff
        precision_argmax = 1.0 * correct_prediction_argmax / total_prediction
        precision_logits = 1.0 * correct_prediction_logits / total_prediction

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision argmax', simple_value=precision_argmax)
        precision_summ.value.add(
            tag='Precision logits', simple_value=precision_logits)
        #summary_writer.add_summary(precision_summ, train_step)
        #  summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision argmax: %.3f, precision logits: %.3f' %
                        (loss, precision_argmax, precision_logits))
        summary_writer.flush()

