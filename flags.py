import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar10', 'mnist, cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('data_path', os.path.join(os.path.expanduser("~"), 'datasets'),
                           'Data dir.')
tf.app.flags.DEFINE_string('model_dir', 'models', 'Directory to keep training outputs.')
tf.app.flags.DEFINE_integer('eval_data_size', 10000,
                            'Size of test set to eval on.')
tf.app.flags.DEFINE_integer('max_steps', 60000,
                            'Max number of steps for training a model.')
tf.app.flags.DEFINE_string('data_dir', 'data',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.model_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
