import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'cifar10', 'mnist, cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('data_path', os.path.join(os.path.expanduser("~"), 'datasets'),
                           'Data dir.')
tf.app.flags.DEFINE_string('models_dir', 'trained_models',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.model_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training.)')
tf.app.flags.DEFINE_integer('min_gpu_number', 0,
                            'Number of gpus used for training. (0 or 1)')
