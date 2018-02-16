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

import cifar_input
import numpy as np
import pixeldp_resnet_conv1
import pixeldp_resnet_img_noise
import pixeldp_cnn_conv1
#  import pixeldp_resnet_img_noise
import tensorflow as tf

import utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
# Download CIFAR10 or CIFAR100 from:
#   e.g. https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tf.app.flags.DEFINE_string('train_data_path',
                           '/home/mathias/data/cifar10_data/cifar-10-batches-bin/data_batch*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path',
                           '/home/mathias/data/cifar10_data/cifar-10-batches-bin/test_batch*',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('model_dir', 'model',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_integer('eval_data_size', 5000,
                            'Size of test set to eval on.')
tf.app.flags.DEFINE_integer('max_steps', 60000,
                            'Max number of steps for training a model.')
tf.app.flags.DEFINE_string('data_dir', 'data',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.model_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')

def evaluate(hps, model, dir_name=None, rerun=False):
    """Evaluate the ResNet and log prediction counters to compute
    sensitivity."""
    if dir_name == None:
        dir_name = FLAGS.data_dir + "/" + FLAGS.model_dir

    if os.path.isfile(dir_name + "/eval_data.json") and not rerun:
        # run only new models
        return

    if FLAGS.dataset == 'mnist':
        mnist   = tf.contrib.learn.datasets.load_dataset("mnist")
        dataset = mnist.test
        images  = tf.placeholder(tf.float32,
                                 [hps.batch_size, 784],
                                 name='x-input')
        labels  = tf.placeholder(tf.int64,
                                 [hps.batch_size],
                                 name='y-input')
    elif FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100':
        images, labels = cifar_input.build_input(
            FLAGS.dataset,
            FLAGS.eval_data_path,
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
            xs, ys = dataset.next_batch(hps.batch_size, fake_data=False)
            args = {model.noise_scale: 1.0,
                    model._images: xs,
                    model._labels: ys}
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

def train(hps, model, dir_name=None):
    """Training loop."""
    if dir_name == None:
        dir_name = FLAGS.data_dir + "/" + FLAGS.model_dir

    if FLAGS.dataset == 'mnist':
        mnist   = tf.contrib.learn.datasets.load_dataset("mnist")
        dataset = mnist.train
        images  = tf.placeholder(tf.float32,
                                 [hps.batch_size, 784],
                                 name='x-input')
        labels  = tf.placeholder(tf.int64,
                                 [hps.batch_size],
                                 name='y-input')
    elif FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100':
        images, labels = cifar_input.build_input(
            FLAGS.dataset,
            FLAGS.eval_data_path,
            hps.batch_size,
            hps.image_standardization,
            FLAGS.mode
        )
    model = model.Model(hps, images, labels, FLAGS.mode)
    model.build_graph()

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

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
                 'precision': precision},
        every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.1
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
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=dir_name,
        hooks=[logging_hook,
               _LearningRateSetterHook(),
               tf.train.StopAtStepHook(last_step=FLAGS.max_steps),],
        chief_only_hooks=[summary_hook],
        # Since we provide a SummarySaverHook, we need to disable default
        # SummarySaverHook. To do that we set save_summaries_steps to 0.
        save_summaries_steps=0,
        config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
      while not mon_sess.should_stop():
        s = 1.0 - min(0.99975**steps, 0.9)
        if s > 0.9: s = 1.0  # this triggers around 10k steps

        if FLAGS.dataset == 'mnist':
            xs, ys = dataset.next_batch(hps.batch_size, fake_data=False)
            args = {model.noise_scale: s,
                    model._images: xs,
                    model._labels: ys}
        elif FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100':
            args = {model.noise_scale: s}

        mon_sess.run(model.train_op, args)
        steps += 1

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
    hps = utils.HParams(batch_size=batch_size,
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
                        noise_scheme='l2_l2_s1',
                        attack_norm_bound=0.3)

    # _model can be:
    #   pixeldp_resnet_conv1 pixeldp_resnet_img_noise
    #   pixeldp_cnn_conv1 pixeldp_cnn_img_noise
    _model = pixeldp_cnn_conv1
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
