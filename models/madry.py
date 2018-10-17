# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, zipfile, hashlib, os, shutil

import numpy as np
import tensorflow as tf

class Model(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.
    Args:
      mode: One of 'train' and 'eval'.
    """

    self.hps  = hps
    self.mode = mode
    self.images = images
    self.labels = labels
    #  self._build_model(x_input, y_input)
    self.noise_scale = None

  def pre_noise_sensitivity(self):
      return None

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def build_graph(self, inputs_tensor=None, labels_tensor=None):
      """Build a whole graph for the model."""
      self.global_step = tf.contrib.framework.get_or_create_global_step()

      #  with tf.variable_scope('model_graph'):
      self._build_model(inputs_tensor, labels_tensor)

  def _build_model(self, inputs_tensor=None, labels_tensor=None):
    """Build the core model within the graph."""

    if inputs_tensor != None:
        self.images = inputs_tensor
    if labels_tensor != None:
        self.labels = labels_tensor

    assert self.mode == 'train' or self.mode == 'eval'

    with tf.variable_scope('input'):

      if self.images == None:
        self.x_input = tf.placeholder(
          tf.float32,
          shape=[None, 32, 32, 3])
      else:
        self.x_input = self.images

      if self.labels is None:
        self.y_input = tf.placeholder(tf.int64, shape=None)
      else:
        self.y_input = tf.argmax(self.labels, 1)

      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))



    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    filters = [16, 160, 320, 640]


    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, 5):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, 5):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, 5):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, 10)
      self.logits = self.pre_softmax

    self.predictions = tf.nn.softmax(self.logits)

    self._predictions = tf.argmax(self.pre_softmax, 1)


    self.correct_prediction = tf.equal(self._predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()
      self.cost = self.mean_xent + self.weight_decay_loss

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
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

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

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
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

  def maybe_download_and_extract(dir_name=None):
    final_dir = os.path.join(dir_name, 'madry')

    if os.path.exists(final_dir):
        return

    # 'adv_trained' model
    url = 'https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip?dl=1'

    # get the name of the file
    zip_dir = url.split('/')[-1].split('?')[0].split('.')[0]

    fname = os.path.join(dir_name, 'madry.zip')

    # model download
    print('Downloading model')
    if sys.version_info >= (3,):
      import urllib.request
      #  urllib.request.urlretrieve(url, fname)
      urllib.request.urlretrieve(url, fname)
    else:
      import urllib
      #  urllib.urlretrieve(url, fname)
      urllib.urlretrieve(url, fname)

    # computing model hash
    sha256 = hashlib.sha256()
    with open(fname, 'rb') as f:
      data = f.read()
      sha256.update(data)
    print('SHA256 hash: {}'.format(sha256.hexdigest()))

    # extracting model
    print('Extracting model')
    with zipfile.ZipFile(fname, 'r') as model_zip:
      model_zip.extractall(dir_name)
      print('Extracted model in {}/{}'.format(dir_name, model_zip.namelist()[0]))
      d = os.path.join(dir_name, model_zip.namelist()[0])

    print('Moving mode to model in {}'.format(final_dir))
    shutil.move(os.path.join(d, zip_dir), final_dir)
    print('Cleaning up')
    os.remove(fname)
    shutil.rmtree(d)

