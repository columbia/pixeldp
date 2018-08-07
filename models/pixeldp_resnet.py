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
from models import pixeldp
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
import math
import utils

from tensorflow.python.training import moving_averages

class Model(pixeldp.Model):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        pixeldp.Model.__init__(self, hps, images, labels, mode)

    def _build_model(self):
        """Build the core model within the graph."""
        assert(self.hps.noise_after_n_layers <= 1)

        with tf.variable_scope('im_dup'):
            # Duplicate images to get multiple draws from the DP label
            # ditribution (each duplicate gets an independent noise draw
            # before going through the rest of the network).
            ones = tf.ones([len(self.images.get_shape())-1], dtype=tf.int32)
            x = tf.tile(self.images, tf.concat([[self.hps.n_draws], ones], axis=0))

        x = self._maybe_add_noise_layer(x, sensitivity_norm=self.hps.sensitivity_norm,
                sensitivity_control_scheme=self.hps.sensitivity_control_scheme,
                position=0)

        with tf.variable_scope('init'):
            filter_size = 3
            in_filters  = x.get_shape()[-1]
            out_filters = 16
            stride      = 1
            strides     = self._stride_arr(stride)

            x = self._conv("init_conv", x, filter_size, in_filters, out_filters,
                    strides, position=1)

        if not self.hps.noise_after_activation:
            x = self._relu(x, self.hps.relu_leakiness)

        x = self._maybe_add_noise_layer(x, sensitivity_norm=self.hps.sensitivity_norm,
                sensitivity_control_scheme=self.hps.sensitivity_control_scheme,
                position=1)

        if self.hps.noise_after_activation:
            x = self._relu(x, self.hps.relu_leakiness)

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual
            #  filters = [16, 16, 32, 64]
            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            filters = [out_filters, 160, 320, 640]
            # Update hps.num_residual_units to 4

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.pre_softmax = logits
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

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm(self, name, x):
      """Batch normalization."""
      with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))

        if self.mode == 'train':
          mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

          moving_mean = tf.get_variable(
              'moving_mean', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32),
              trainable=False)
          moving_variance = tf.get_variable(
              'moving_variance', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32),
              trainable=False)

          self._extra_train_ops.append(moving_averages.assign_moving_average(
              moving_mean, mean, 0.9))
          self._extra_train_ops.append(moving_averages.assign_moving_average(
              moving_variance, variance, 0.9))
        else:
          mean = tf.get_variable(
              'moving_mean', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32),
              trainable=False)
          variance = tf.get_variable(
              'moving_variance', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32),
              trainable=False)
          tf.summary.histogram(mean.op.name, mean)
          tf.summary.histogram(variance.op.name, variance)
        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
      """Residual unit with 2 sub layers."""
      if activate_before_residual:
        with tf.variable_scope('shared_activation'):
          x = self._batch_norm('init_bn', x)
          x = self._relu(x, self.hps.relu_leakiness)
          orig_x = x
      else:
        with tf.variable_scope('residual_only_activation'):
          orig_x = x
          x = self._batch_norm('init_bn', x)
          x = self._relu(x, self.hps.relu_leakiness)

      with tf.variable_scope('sub1'):
        x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = self._batch_norm('bn2', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

      with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
          orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
          orig_x = tf.pad(
              orig_x, [[0, 0], [0, 0], [0, 0],
                       [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        x += orig_x

      tf.logging.debug('image after unit %s', x.get_shape())
      return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
      """Bottleneck residual unit with 3 sub layers."""
      if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
          x = self._batch_norm('init_bn', x)
          x = self._relu(x, self.hps.relu_leakiness)
          orig_x = x
      else:
        with tf.variable_scope('residual_bn_relu'):
          orig_x = x
          x = self._batch_norm('init_bn', x)
          x = self._relu(x, self.hps.relu_leakiness)

      with tf.variable_scope('sub1'):
        x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

      with tf.variable_scope('sub2'):
        x = self._batch_norm('bn2', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

      with tf.variable_scope('sub3'):
        x = self._batch_norm('bn3', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

      with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
          orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
        x += orig_x

      tf.logging.info('image after unit %s', x.get_shape())
      return x

    def _global_avg_pool(self, x):
      assert x.get_shape().ndims == 4
      return tf.reduce_mean(x, [1, 2])
