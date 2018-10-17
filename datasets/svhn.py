from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys, glob

import numpy as np
from scipy.io import loadmat

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('aug_trans', False, "")
tf.app.flags.DEFINE_bool('aug_flip', False, "")

NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_images_and_labels(images, labels, filepath):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image = images[index].tolist()
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(32),
            'width': _int64_feature(32),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[index])),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()


def read(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([3072], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    image = features['image']
    image = tf.reshape(image, [32, 32, 3])
    label = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
    return image, label


def generate_batch(
        example,
        min_queue_examples,
        batch_size, mode):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1

    if mode == 'train':
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=False,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=False,
            capacity=min_queue_examples + 3 * batch_size)

    return ret

def transform(image):
    image = tf.reshape(image, [32, 32, 3])
    if FLAGS.aug_trans or FLAGS.aug_flip:
        print("augmentation")
        if FLAGS.aug_trans:
            image = tf.pad(image, [[2, 2], [2, 2], [0, 0]])
            image = tf.random_crop(image, [32, 32, 3])
        if FLAGS.aug_flip:
            image = tf.image.random_flip_left_right(image)
    return image

def generate_filename_queue(filenames, svhn_dir, num_epochs=None):
    print("filenames in queue:", filenames)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(svhn_dir, filenames[i])
    return tf.train.string_input_producer(filenames, num_epochs=num_epochs)


def maybe_download_and_extract(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    filepath_train_mat = os.path.join(data_path, 'train_32x32.mat')
    filepath_test_mat = os.path.join(data_path, 'test_32x32.mat')
    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

    # Training set
    print("Loading training data...")
    print("Preprocessing training data...")
    train_data = loadmat(data_path + '/train_32x32.mat')
    train_x = (-127.5 + train_data['X']) / 255.
    train_x = train_x.transpose((3, 0, 1, 2))
    train_x = train_x.reshape([train_x.shape[0], -1])
    train_y = train_data['y'].flatten().astype(np.int32)
    train_y[train_y == 10] = 0

    # Test set
    print("Loading test data...")
    test_data = loadmat(data_path + '/test_32x32.mat')
    test_x = (-127.5 + test_data['X']) / 255.
    test_x = test_x.transpose((3, 0, 1, 2))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = test_data['y'].flatten().astype(np.int32)
    test_y[test_y == 10] = 0

    np.save('{}/train_images'.format(data_path), train_x)
    np.save('{}/train_labels'.format(data_path), train_y)
    np.save('{}/test_images'.format(data_path), test_x)
    np.save('{}/test_labels'.format(data_path), test_y)


def load_svhn(data_path):
    maybe_download_and_extract(data_path)
    train_images = np.load('{}/train_images.npy'.format(data_path)).astype(np.float32)
    train_labels = np.load('{}/train_labels.npy'.format(data_path)).astype(np.float32)
    test_images = np.load('{}/test_images.npy'.format(data_path)).astype(np.float32)
    test_labels = np.load('{}/test_labels.npy'.format(data_path)).astype(np.float32)
    return (train_images, train_labels), (test_images, test_labels)


def prepare_dataset(data_path):
    (train_images, train_labels), (test_images, test_labels) = load_svhn(data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    convert_images_and_labels(train_images, train_labels,
                              os.path.join(data_path, 'train.tfrecords'))
    convert_images_and_labels(test_images, test_labels,
                              os.path.join(data_path, 'test.tfrecords'))

def build_input(data_path, batch_size, image_standardization, mode):

    assert(not image_standardization)  # not supported for now
    data_path = os.path.join(data_path, 'svhn')

    if mode == 'train':
        filenames = ['train.tfrecords']
        num_examples = NUM_EXAMPLES_TRAIN
    elif mode == 'eval':
        filenames = ['test.tfrecords']
        num_examples = NUM_EXAMPLES_TEST

    prepared = True
    for filename in filenames:
        fname = os.path.join(data_path, filename)
        if not os.path.exists(fname):
            prepared = False
    if not prepared:
        prepare_dataset(data_path)

    filename_queue = generate_filename_queue(filenames, data_path)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32)) if mode == 'train' else image
    return generate_batch([image, label], num_examples, batch_size, mode)

