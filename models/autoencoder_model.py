# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np
import tensorflow as tf
import six
import math

from tensorflow.python.training import moving_averages


def l1_normalize(x, dim, epsilon=1e-12, name=None):
  """Normalizes along dimension `dim` using an L2 norm.
  For a 1-D tensor with `dim = 0`, computes
      output = x / max(sum(abs(x)), epsilon)
  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `dim`.
  Args:
    x: A `Tensor`.
    dim: Dimension along which to normalize.  A scalar or a vector of
      integers.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).
  Returns:
    A `Tensor` with the same shape as `x`.
  """
  with tf.name_scope(name, "l1_normalize", [x]) as name:
    x          = tf.convert_to_tensor(x, name            = "x")
    abs_sum    = tf.reduce_sum(tf.abs(x), dim, keep_dims = True)
    x_inv_norm = tf.reciprocal(tf.maximum(abs_sum, epsilon))
    return tf.multiply(x, x_inv_norm, name=name)


class Autoencoder(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        assert(hps.noise_scheme in["l2", "l1", "vanilla", "l1_l1_s1", "l1_l2", "l1_l2_s1",
                                   "l1_l1", "l2_l2_s1", "linf_linf_s1",
                                   "linf_linf_l2_s1"])
        self.hps = hps
        self.mode = mode
        self.labels = labels
        self._images = images

        with tf.variable_scope('im_dup'):
            ones = tf.ones([len(self._images.get_shape())-1], dtype=tf.int32)
            self._images = tf.tile(self._images,
                                   tf.concat([[self.hps.n_draws], ones], axis=0))
        try:
            self.use_pooling = hps.use_pooling
        except Exception:
            self.use_pooling = False

        try:
            self.use_batchnorm = hps.use_batchnorm
        except Exception:
            self.use_batchnorm = False


        self._epsilon_dp = 1.0
        self._image_size = 32
        self._delta_dp   = 0.05

        self._parseval_convs  = []
        self._parseval_ws     = []
        self._extra_train_ops = []

        self.test = None

        self._pre_noise_Ws                 = []
        self._pre_noise_bs                 = []
        self._pre_noise_strides            = []
        self._pre_noise_layers             = []
        self._pre_noise_l2_sensitivities   = []
        self._pre_noise_linf_sensitivities = []

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

    def _build_model(self):
        """Build the core model within the graph."""
        shapes = []
        encoder = []
        output_shapes = []

        # use sigmoid -- if applicable
        use_sigmoid = False
        try:
            use_sigmoid = self.hps.use_sigmoid
        except Exception:
            pass

        filter_sizes = self.hps.filter_sizes
        n_filters = self.hps.n_filters
        strides = self.hps.strides

        self.noise_scale = tf.placeholder(tf.float32, shape=(),
                                          name='noise_scale')
        # handle image noise
        x = self._images

        if self.hps.noise_placement == 'img_noise':
            self.l1_mean_signal = tf.reduce_mean(tf.norm(x, ord=1))

            self.l2_mean_signal = tf.reduce_mean(tf.norm(x))
            noise = self.img_dp_noise(x)
            self.l1_mean_noise = tf.reduce_mean(tf.norm(noise, ord=1))
            self.l2_mean_noise = tf.reduce_mean(tf.norm(noise))
            x += noise

        current_input = x
        for layer_i, n_output in enumerate(n_filters):
            n_input = current_input.get_shape().as_list()[3]

            # initialize encoder kernels per layer
            k = filter_sizes[layer_i]
            s = strides[layer_i][1]
            W = tf.Variable(
                tf.random_normal(
                    [k, k, n_input, n_output],
                    mean=.0,
                    stddev= 2.0 / (filter_sizes[layer_i]**2 * n_output)
                ),
                name='DW-encoder-{}'.format(layer_i)
            )
            # normalize before first convolution -- if applicable
            b = tf.Variable(tf.ones([n_output]),
                            name='b-encoder-{}'.format(layer_i))

            if self.use_batchnorm and layer_i > self.hps.noise_placement_layer:
                current_input = self._batch_norm(current_input,
                                                 "encoder-{}".format(layer_i))

            if self.hps.noise_placement == 'conv_noise' and  \
                    layer_i <= self.hps.noise_placement_layer:
                W = self.normalize_kernel(W)
            encoder.append(W)

            output = tf.add(
                tf.nn.conv2d(current_input, W, strides=strides[layer_i], padding='SAME'), b
            )
            if layer_i <= self.hps.noise_placement_layer:
                self._pre_noise_layer = output

                self._pre_noise_strides.append(strides[layer_i])
                self._pre_noise_Ws.append(W)
                self._pre_noise_bs.append(b)

                self._pre_noise_layers.append(output)
                #  svd = tf.svd(W, compute_uv=False)
                shape = W.get_shape().as_list()
                conv_mult = float(math.ceil(k/s))
                w_t = tf.reshape(W * conv_mult, [-1, shape[-1]])
                w   = tf.transpose(w_t)
                svd = tf.svd(w, compute_uv=False)
                self._pre_noise_l2_sensitivities.append(svd)
                l1s = tf.reduce_sum(tf.abs(W), [0, 1, 2], keep_dims = True)
                self._pre_noise_linf_sensitivities.append(l1s)

            # add noise after first convolution -- if applicable
            if self.hps.noise_placement == 'conv_noise' and  \
                    layer_i == self.hps.noise_placement_layer:
                try:
                    a = self.hps.noise_after_activation
                except Exception:
                    self.hps.noise_after_activation = True

                if self.hps.noise_after_activation:
                    output = self._relu(output, self.hps.relu_leakiness)

                self.l1_mean_signal = tf.reduce_mean(tf.norm(output, ord=1))
                self.l2_mean_signal = tf.reduce_mean(tf.norm(output))

                noise  = self.conv_dp_noise(output, W, k, s)
                self.l1_mean_noise = tf.reduce_mean(tf.norm(noise, ord=1))
                self.l2_mean_noise = tf.reduce_mean(tf.norm(noise))

                if self.hps.noise_after_activation:
                    output += noise
                else:
                    output += noise
                    output = self._relu(output, self.hps.relu_leakiness)

            elif layer_i  == len(n_filters)-1 and use_sigmoid:
                output = tf.nn.sigmoid(output)
            elif layer_i > 0:
                output = self._relu(output, self.hps.relu_leakiness)

            if self.use_pooling:
                output = tf.nn.avg_pool(output, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding='SAME')

            shapes.append(current_input.get_shape().as_list())
            output_shapes.append(output.get_shape().as_list())
            current_input = output

        self.encoder = encoder[0]

        # reverse for decoder
        shapes.reverse()
        encoder.reverse()
        n_filters.reverse()
        filter_sizes.reverse()
        strides.reverse()
        print("\n\nLatent Representation:", current_input, "\n\n")

        # Build the decoder using the same weights
        for layer_i, shape in enumerate(shapes):

            if self.use_batchnorm:
                current_input = self._batch_norm(current_input,
                                                 "decoder-{}".format(layer_i))
            if self.use_pooling:
                current_input = tf.image.resize_nearest_neighbor(current_input,
                                                                 shape[1:3])
            # decoder kernels (transpose encoder in case of tied weights)
            if self.hps.tied_weigth:
                W = encoder[layer_i]
            else:
                W = tf.Variable(
                    tf.random_normal(
                        [
                            filter_sizes[layer_i],
                            filter_sizes[layer_i],
                            shapes[layer_i][3],
                            n_filters[layer_i]
                        ],
                        mean=.0,
                        stddev= 2.0 / (filter_sizes[layer_i]**2 * n_output)
                    ),
                    name='DW-decoder-{}'.format(layer_i)
                )
            b = tf.Variable(tf.ones([W.get_shape().as_list()[2]]),
                            name='b-decoder-{}'.format(layer_i))

            if layer_i == len(shapes) - 1:
                # transpose convolve and activation
                output = tf.nn.sigmoid(
                    tf.add(
                        tf.nn.conv2d_transpose(
                            current_input,
                            W,
                            tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                            strides=strides[layer_i],
                            padding='SAME'
                        ),
                        b
                    )
                )
            else:
                # transpose convolve and activation
                output = self._relu(
                    tf.add(
                        tf.nn.conv2d_transpose(
                            current_input,
                            W,
                            tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                            strides=strides[layer_i],
                            padding='SAME'
                        ),
                        b
                    )
                )

            current_input = output

        self.output = output

        # RMSE
        with tf.variable_scope('costs'):
          self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self._images,
                                                                   output))))
          # self.cost += self._decay()
          tf.summary.scalar('cost', self.cost)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.placeholder(tf.float32, shape=(),
                                       name='learning_rate')
        if self.hps.optimizer == 'sgd':
          optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
          optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
          optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'adadelta':
          optimizer = tf.train.AdadeltaOptimizer(self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        accum_vars = [
            tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
            for tv in trainable_variables
        ]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        self.grads = optimizer.compute_gradients(self.cost, trainable_variables)

        self.accum_ops = [
            accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.grads)
        ]

        apply_op = optimizer.apply_gradients(
            [(accum_vars[i], gv[1]) for i, gv in enumerate(self.grads)],
            global_step=self.global_step
        )

        train_ops = [apply_op] + self._extra_train_ops
        previous_ops = [tf.group(*train_ops)]
        if self.hps.noise_scheme == 'l2_l2_s1':
            # Parseval
            with tf.control_dependencies(previous_ops):
                parseval_update = tf.group(*self._build_parseval_update_ops())
                previous_ops    = [parseval_update]

        with tf.control_dependencies(previous_ops):
            self.train_op = tf.no_op(name='train')

    def _relu(self, x, leakiness=0.0):
      """Relu, with optional leaky support."""
      return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name[:2] == 'DW':
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _dp_mult(self, size=None):
        epsilon_dp = self._epsilon_dp
        delta_dp   = self._delta_dp
        max_pixeldp_norm = self.hps.pixeldp_norm_bound
        if self.hps.noise_scheme == 'l1_l2'    or  \
            self.hps.noise_scheme == 'l1_l2_s1' or  \
            self.hps.noise_scheme == 'l2_l2_s1' or\
            self.hps.noise_scheme == 'l2':
            return max_pixeldp_norm * \
                math.sqrt(2 * math.log(1.25 / delta_dp)) / epsilon_dp
        elif self.hps.noise_scheme == 'l1_l1'  or  \
            self.hps.noise_scheme == 'l1_l1_s1' or\
            self.hps.noise_scheme == 'l1':
            return max_pixeldp_norm / epsilon_dp
        elif self.hps.noise_scheme == 'linf_linf_l2_s1':
            return max_pixeldp_norm * \
                math.sqrt(2 * size * math.log(1.25 / delta_dp)) / epsilon_dp
        elif self.hps.noise_scheme == 'linf_linf_s1':
            return 2 * max_pixeldp_norm / epsilon_dp
        elif self.hps.noise_scheme == 'vanilla':
            return .0
        else:
            print("What noise is this? Dah")
            assert(False)

    def conv_dp_noise(self, x, kernel, k, s):
        if self.hps.noise_scheme == 'l1_l1':
            self.l1_norms = tf.norm(
              tf.reshape(tf.stack(tf.unstack(kernel, axis=2)), shape=[3, -1]),
              ord=1,
              axis=1
            )
            self.norms = self.l1_norms
            self.sensitivity_multiplier = tf.reduce_max(self.norms)

            dp_mult          = self._dp_mult()

            laplace_shape = tf.shape(x)
            loc           = tf.zeros(laplace_shape, dtype=tf.float32)
            scale         = tf.ones(laplace_shape,  dtype=tf.float32)
            epsilon       = tf.distributions.Laplace(loc, scale).sample()

            self.l1_sensitivity = tf.reduce_max(self.l1_norms)
            self.b              = self.noise_scale * dp_mult * self.l1_sensitivity

            noise   = self.b * epsilon
        elif self.hps.noise_scheme == 'l1_l2':
            self.l2_norms = tf.norm(
              tf.reshape(tf.stack(tf.unstack(kernel, axis=2)), shape=[3, -1]),
              ord=2,
              axis=1
            )
            self.norms = self.l2_norms
            self.sensitivity_multiplier = tf.reduce_max(self.norms)

            dp_mult     = self._dp_mult()

            epsilon        = tf.random_normal(tf.shape(x), mean=0, stddev=1)
            self.l2_sensitivity = tf.reduce_max(self.l2_norms)
            self.sigma          = tf.multiply(dp_mult, self.l2_sensitivity, name='noise_scale')

            self.noise_stddev = self.noise_scale  * self.sigma
            noise   = self.noise_stddev * epsilon
        elif self.hps.noise_scheme == 'l1_l1_s1':
            # We compute norms just for logging, they are then normalized
            # before being used in the Convolution.
            self.l1_norms = tf.norm(
              tf.reshape(tf.stack(tf.unstack(kernel, axis=2)), shape=[3, -1]),
              ord=1,
              axis=1
            )
            self.norms = self.l1_norms
            self.sensitivity_multiplier = tf.reduce_max(self.norms)

            dp_mult          = self._dp_mult()

            print(dp_mult)
            laplace_shape = tf.shape(x)
            loc           = tf.zeros(laplace_shape, dtype=tf.float32)
            scale         = tf.ones(laplace_shape,  dtype=tf.float32)
            epsilon       = tf.distributions.Laplace(loc, scale).sample()

            self.l1_sensitivity = 1.0  # cause we normalize
            self.b              = self.noise_scale * dp_mult * self.l1_sensitivity
            print(self.b)

            noise   = self.b * epsilon
        elif self.hps.noise_scheme == 'l1_l2_s1':
            # We compute norms just for logging, they are then normalized
            # before being used in the Convolution.
            self.l2_norms = tf.norm(
              tf.reshape(tf.stack(tf.unstack(kernel, axis=2)), shape=[3, -1]),
              ord=2,
              axis=1
            )
            self.norms = self.l2_norms
            self.sensitivity_multiplier = tf.reduce_max(self.norms)

            dp_mult     = self._dp_mult()

            epsilon        = tf.random_normal(tf.shape(x), mean=0, stddev=1)
            self.l2_sensitivity = 1.0
            self.sigma          = tf.multiply(dp_mult, self.l2_sensitivity, name='noise_scale')

            self.noise_stddev = self.noise_scale  * self.sigma
            noise   = self.noise_stddev * epsilon
        elif self.hps.noise_scheme == 'l2_l2_s1':
            shape      = kernel.get_shape().as_list()
            # We need to multiply by k to get the true sensitivity
            conv_mult  = float(math.ceil(k/s))
            w_t        = tf.reshape(kernel*conv_mult, [-1, shape[-1]])
            w          = tf.transpose(w_t)
            self.norms = tf.svd(w, compute_uv=False)
            self.sensitivity_multiplier = tf.reduce_max(self.norms)

            dp_mult     = self._dp_mult()

            epsilon        = tf.random_normal(tf.shape(x), mean=0, stddev=1)
            self.l2_sensitivity = 1.0
            self.sigma          = tf.multiply(dp_mult, self.l2_sensitivity, name='noise_scale')

            self.noise_stddev = self.noise_scale  * self.sigma
            noise   = self.noise_stddev * epsilon
        elif self.hps.noise_scheme == 'linf_linf_l2_s1':
            # We compute norms just for logging, they are then normalized
            # before being used in the Convolution.
            self.l1_norms = x
            self.norms = self.l1_norms

            dp_mult     = self._dp_mult(size=np.prod(x.get_shape().as_list()[1:]))

            epsilon     = tf.random_normal(tf.shape(x), mean=0, stddev=1)
            self.sensitivity = 1.0
            self.sigma       = tf.multiply(dp_mult, self.sensitivity, name='noise_scale')

            self.noise_stddev = self.noise_scale  * self.sigma
            self.sensitivity_multiplier = self.noise_stddev  # for logging
            noise   = self.noise_stddev * epsilon
        elif self.hps.noise_scheme == 'linf_linf_s1':
            # We compute norms just for logging, they are then normalized
            # before being used in the Convolution.
            self.norms = x

            print(x.get_shape().as_list())
            print(np.prod(x.get_shape().as_list()[1:]))
            n       = tf.cast(np.prod(x.get_shape().as_list()[1:]), tf.float32)
            dp_mult = self._dp_mult()
            print(dp_mult)
            self.sensitivity = dp_mult / 1.0

            _shape = tf.shape(x)
            l_inf   = tf.distributions.Gamma(n, 1/(dp_mult*self.noise_scale)).sample()
            self.sensitivity_multiplier = l_inf  # for logging
            size    = l_inf * tf.ones(_shape,  dtype=tf.float32)
            # TODO: one of these numbers has to be exactly l_inf FIXME
            epsilon = tf.distributions.Uniform(low=-size, high=size).sample()

            noise   = epsilon
        elif self.hps.noise_scheme == 'vanilla':
            noise = x
            self.sensitivity_multiplier = tf.reduce_max(x)
        else:
            print("What noise is this? Dah")
            assert(False)

        return noise

    def img_dp_noise(self, x):

        if self.hps.noise_scheme == 'l1':
            self.sensitivity_multiplier = epsilon[0][0][0][0] # just to not break main...
            dp_mult          = self._dp_mult()

            laplace_shape = tf.shape(x)
            loc           = tf.zeros(laplace_shape, dtype=tf.float32)
            scale         = tf.ones(laplace_shape,  dtype=tf.float32)
            epsilon       = tf.distributions.Laplace(loc, scale).sample()

            self.l1_sensitivity = 1.0  # pixel size
            self.b              = self.noise_scale * dp_mult * self.l1_sensitivity
            noise   = self.b * epsilon
        elif self.hps.noise_scheme == 'l2':
            dp_mult     = self._dp_mult()
            epsilon             = tf.random_normal(tf.shape(x), mean=0, stddev=1)
            self.sensitivity_multiplier = epsilon[0][0][0][0] # just to not break main...
            self.l2_sensitivity = 1.0
            self.sigma          = tf.multiply(dp_mult, self.l2_sensitivity,
                                              name='noise_scale')

            self.noise_stddev = self.noise_scale  * self.sigma
            noise   = self.noise_stddev * epsilon
        elif self.hps.noise_scheme == 'vanilla':
            noise = x
        else:
            print("What noise is this? Dah")
            assert(False)

        return noise

    def normalize_kernel(self, kernel):
        # Change stride
        if self.hps.noise_scheme == 'l1_l1_s1':
            # Sensitivity 1 by L1 normalization
            k = utils.l1_normalize(kernel, dim=[0, 1, 3])
        elif self.hps.noise_scheme == 'l1_l2_s1':
            # Sensitivity 1 by L2 normalization
            k = tf.nn.l2_normalize(kernel, dim=[0, 1, 3])
        elif self.hps.noise_scheme == 'l2_l2_s1':
            # Parseval
            self._parseval_convs.append(kernel)
            assert(self.hps.strides[0][1] == self.hps.strides[0][2])
            conv_mult = math.ceil(self.hps.filter_sizes[0]/self.hps.strides[0][1])
            k = kernel / float(conv_mult)
        elif self.hps.noise_scheme == 'linf_linf_l2_s1' or  \
             self.hps.noise_scheme == 'linf_linf_s1':
            # Sensitivity 1 by L1 normalization
            k = utils.l1_normalize(kernel, dim=[0, 1, 2])
        else:
            k = kernel
        return k

    def _build_parseval_update_ops(self):
        """_build_parseval_update_ops"""
        beta = 0.05

        ops  = []

        for kernel in self._parseval_convs:
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


    def _batch_norm(self, x, scope):
        with tf.variable_scope("batchnorm_{}".format(scope)):
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
