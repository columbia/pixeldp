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

"""ResNet model.
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
import math
import utils

from tensorflow.python.training import moving_averages

class Model(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps

        self._images = images
        self._labels = labels
        self.mode    = mode

        # Differential Privacy parameters
        self._epsilon_dp = 1.0
        self._delta_dp   = 0.05

        # Differential Privacy parameters
        self._image_size = self.hps.image_size

        # Extra book keeping for Parseval
        self._parseval_convs  = []
        self._parseval_ws     = []
        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _dp_mult(self, size=None):
       epsilon_dp = self._epsilon_dp
       delta_dp   = self._delta_dp
       max_attack_norm = self.hps.attack_norm_bound
       if self.hps.noise_scheme == 'l1_l2'    or  \
          self.hps.noise_scheme == 'l1_l2_s1' or  \
          self.hps.noise_scheme == 'l2_l2_s1':
          # Use the Gaussian mechanism
          return max_attack_norm * math.sqrt(2 * math.log(1.25 / delta_dp)) / epsilon_dp
       elif self.hps.noise_scheme == 'l1_l1'  or  \
            self.hps.noise_scheme == 'l1_l1_s1':
          # Use the Laplace mechanism
          return max_attack_norm / epsilon_dp
       else:
          return 0

    def _build_model(self):
        """Build the core model within the graph."""
        # TODO: make this outside the CNN, it's MNIST specific
        input_layer = tf.reshape(self._images, [-1, 28, 28, 1])
        self.labels = tf.one_hot(self._labels, self.hps.num_classes)

        with tf.variable_scope('im_dup'):
            # Duplicate images to get multiple draws from the DP label
            # ditribution (each duplicate gets an independent noise draw
            # before going through the rest of the network).
            ones = tf.ones([len(input_layer.get_shape())-1], dtype=tf.int32)
            x = tf.tile(input_layer, tf.concat([[self.hps.n_draws], ones], axis=0))

        with tf.variable_scope('init'):
            with tf.variable_scope('init_conv'):
                filter_size = 5
                in_filters  = 1
                out_filters = 32
                stride      = 2
                strides     = self._stride_arr(stride)
                n = filter_size * filter_size * out_filters
                self.kernel = tf.get_variable(
                  'DW',
                  [filter_size, filter_size, in_filters, out_filters],
                  tf.float32,
                  initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
                )

                if self.hps.noise_scheme == 'l2_l2_s1':
                    # Parseval projection, see: https://arxiv.org/abs/1704.08847
                    self._parseval_convs.append(self.kernel)
                    sensitivity_rescaling = math.ceil(stride / filter_size)
                    k = sensitivity_rescaling * self.kernel
                elif self.hps.noise_scheme == 'l1_l2_s1':
                    # Sensitivity 1 by L2 normalization
                    k = tf.nn.l2_normalize(self.kernel, dim=[0, 1, 3])
                elif self.hps.noise_scheme == 'l1_l1_s1':
                    # Sensitivity 1 by L1 normalization
                    k = utils.l1_normalize(self.kernel, dim=[0, 1, 3])
                else:
                    k = self.kernel

                x = tf.nn.conv2d(x, k, strides, padding='SAME')

            ############
            # DP noise #

            # This is a factor applied to the noise layer,
            # used to rampup the noise at the beginning of training.
            self.noise_scale = tf.placeholder(tf.float32, shape=(), name='noise_scale')

            if self.hps.noise_scheme == 'l1_l2':
                sqr_sum       = tf.reduce_sum(tf.square(x), [0, 1, 3],
                                              keep_dims=True)
                self.l2_norms = tf.sqrt(sqr_sum)

                dp_mult          = self._dp_mult()
                epsilon          = tf.random_normal(tf.shape(x), mean=0, stddev=1)
                self.sensitivity = tf.reduce_max(self.l2_norms)
                self.sigma       = tf.multiply(dp_mult, self.sensitivity)

                self.noise_stddev = self.noise_scale  * self.sigma
                self.noise        = self.noise_stddev * epsilon
                x                 = x + self.noise
            elif self.hps.noise_scheme == 'l1_l2_s1':
                dp_mult          = self._dp_mult()
                epsilon          = tf.random_normal(tf.shape(x), mean=0, stddev=1)
                self.sensitivity = 1.0  # we bound it
                self.sigma       = tf.multiply(dp_mult, self.sensitivity)

                self.noise_stddev = self.noise_scale  * self.sigma
                self.noise        = self.noise_stddev * epsilon
                x                 = x + self.noise
            elif self.hps.noise_scheme == 'l2_l2_s1':
                # Compute the actual sensitivity to rescale later
                shape      = self.kernel.get_shape().as_list()
                w_t        = tf.reshape(self.kernel, [-1, shape[-1]])
                w          = tf.transpose(w_t)
                self.norms = tf.svd(w, compute_uv=False)
                self.sensitivity_multiplier = tf.reduce_max(self.norms)
                #

                dp_mult          = self._dp_mult()
                epsilon          = tf.random_normal(tf.shape(x), mean=0, stddev=1)
                self.sensitivity = 1.0
                self.sigma       = tf.multiply(dp_mult, self.sensitivity)

                self.noise_stddev = self.noise_scale  * self.sigma
                self.noise        = self.noise_stddev * epsilon
                x                 = x + self.noise
            elif self.hps.noise_scheme == 'l1_l1':
                self.l1_norms = tf.reduce_sum(tf.abs(x), [0, 1, 3],
                                              keep_dims=True)

                dp_mult       = self._dp_mult()
                laplace_shape = tf.shape(x)
                loc           = tf.zeros(laplace_shape, dtype=tf.float32)
                scale         = tf.ones(laplace_shape,  dtype=tf.float32)
                epsilon       = tf.distributions.Laplace(loc, scale).sample()

                self.sensitivity = tf.reduce_max(self.l1_norms)
                self.b           = self.noise_scale * dp_mult * self.sensitivity

                self.noise   = self.b * epsilon
                x            = x + self.noise
            elif self.hps.noise_scheme == 'l1_l1_s1':
                dp_mult       = self._dp_mult()
                laplace_shape = tf.shape(x)
                loc           = tf.zeros(laplace_shape, dtype=tf.float32)
                scale         = tf.ones(laplace_shape,  dtype=tf.float32)
                epsilon       = tf.distributions.Laplace(loc, scale).sample()

                self.sensitivity = 1.0  # because we normalize
                self.b           = self.noise_scale * dp_mult * self.sensitivity

                self.noise   = self.b * epsilon
                x            = x + self.noise
            # DP noise #
            ############

        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv("conv2", x, 5, out_filters, 64, self._stride_arr(2))
        x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('dense'):
            x = self._fully_connected(x, 1024)
            x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('label_dup'):
            ones   = tf.ones([len(self.labels.get_shape())-1], dtype=tf.int32)
            labels = tf.tile(self.labels, tf.concat([[self.hps.n_draws], ones], axis=0))

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)

    def _build_parseval_update_ops(self):
          beta = 0.001

          ops  = []
          for kernel in self._parseval_convs:
              #  shape=[3, 3, 3, 16]
              shape = kernel.get_shape().as_list()

              w_t        = tf.reshape(kernel, [-1, shape[-1]])
              w          = tf.transpose(w_t)
              parseval_k = (1 + beta) * w - beta * tf.matmul(w, tf.matmul(w_t, w))

              op = tf.assign(kernel,
                             tf.reshape(tf.transpose(parseval_k), shape),
                             validate_shape=True)

              ops.append(op)

          for w_t in self._parseval_ws:
              w = tf.transpose(w_t)
              parseval_w = (1 + beta) * w - beta * tf.matmul(w, tf.matmul(w_t, w))
              op = tf.assign(w_t, tf.transpose(parseval_w), validate_shape=True)
              ops.append(op)

          return ops

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops

        previous_ops = [tf.group(*train_ops)]
        if self.hps.noise_scheme == 'l2_l2_s1':
            # Parseval
            with tf.control_dependencies(previous_ops):
                parseval_update = tf.group(*self._build_parseval_update_ops())
                previous_ops    = [parseval_update]

        with tf.control_dependencies(previous_ops):
            self.train_op = tf.no_op(name='train')

    def _decay(self):
      """L2 weight decay loss."""
      costs = []
      for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
          costs.append(tf.nn.l2_loss(var))
          # tf.summary.histogram(var.op.name, var)

      return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
      """Convolution."""
      with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
      """Relu, with optional leaky support."""
      return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
      """FullyConnected layer for final output."""
      x = tf.reshape(x, [self.hps.batch_size * self.hps.n_draws, -1])
      w = tf.get_variable(
          'DW', [x.get_shape()[1], out_dim],
          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      b = tf.get_variable('biases', [out_dim],
                          initializer=tf.constant_initializer())
      return tf.nn.xw_plus_b(x, w, b)
