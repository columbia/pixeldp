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

from flags import FLAGS

def train(hps, model, dataset=None, dir_name=None, dev='/cpu:0'):
    """Training loop."""

    # Trick to start from arbitrary GPU  number
    gpu = int(dev.split(":")[1]) + FLAGS.min_gpu_number
    if gpu >= 16:
        gpu -= 16
    dev = "{}:{}".format(dev.split(":")[0], gpu)


    print("Trainning model on dev:{}".format(dev))
    with tf.device(dev):
        if dir_name == None:
            dir_name = FLAGS.models_dir

        dir_name = os.path.join(dir_name, models.params.name_from_params(model, hps))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(os.path.join(dir_name, 'params.json'), 'w') as f:
            f.write(json.dumps(hps._asdict()))

        if dataset == None:
            dataset = FLAGS.dataset

        images, labels = datasets.build_input(
            dataset,
            FLAGS.data_path,
            hps.batch_size,
            hps.image_standardization,
            'train'
        )
        model = model.Model(hps, images, labels, 'train')
        model.build_graph()

        # param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        #     tf.get_default_graph(),
        #     tfprof_options=tf.contrib.tfprof.model_analyzer.
        #         TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        # sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        # tf.contrib.tfprof.model_analyzer.print_model_analysis(
        #     tf.get_default_graph(),
        #     tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

        truth         = tf.argmax(model.labels, axis=1)
        predictions   = tf.argmax(model.predictions, axis=1)
        one_hot_preds = tf.one_hot(predictions, depth=hps.num_classes, dtype=tf.float32)
        votes         = tf.reshape(one_hot_preds, [hps.n_draws, hps.batch_size, hps.num_classes])
        predictions   = tf.argmax(tf.reduce_sum(votes, axis=0), axis=1)
        precision     = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

        summary_hook = tf.train.SummarySaverHook(
            save_steps=100,
            output_dir=dir_name,
            summary_op=tf.summary.merge([model.summaries,
                                         tf.summary.scalar('Precision', precision)]))

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': model.global_step,
                     'loss': model.cost,
                     'precision': precision,
                     'sensitivity': model.pre_noise_sensitivity()},
            every_n_iter=100)

        class _LearningRateSetterHook(tf.train.SessionRunHook):
            """Sets learning_rate based on global step."""

            def begin(self):
                self._lrn_rate = hps.lrn_rate
                self._schedule = list(zip(hps.lrn_rte_changes, hps.lrn_rte_vals))

            def before_run(self, run_context):
                return tf.train.SessionRunArgs(
                    model.global_step,  # Asks for global step value.
                    feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

            def after_run(self, run_context, run_values):
                train_step = run_values.results
                if len(self._schedule) > 0 and train_step >= self._schedule[0][0]:
                    # Update learning rate according to the schedule.
                    self._lrn_rate = self._schedule[0][1]
                    self._schedule = self._schedule[1:]

        print("START TRAINING")
        steps       = 0

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = str(dev.split(":")[-1])
        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=dir_name,
            hooks=[logging_hook,
                   _LearningRateSetterHook(),
                   tf.train.StopAtStepHook(last_step=hps.steps_num),],
            chief_only_hooks=[summary_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            config=config) as mon_sess:

          ckpt = tf.train.get_checkpoint_state(dir_name)
          if ckpt and ckpt.model_checkpoint_path:
            steps = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

          while not mon_sess.should_stop() and steps < hps.steps_num:
            s = 1.0 - min(0.99975**steps, 0.9)
            if s > 0.9: s = 1.0  # this triggers around 10k steps

            args = {model.noise_scale: s}
            mon_sess.run(model.train_op, args)
            steps += 1

