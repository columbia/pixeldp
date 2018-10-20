# Copyright (C) 2018 Vaggelis Atlidakis <vatlidak@cs.columbia.edu> and
# Mathias Lecuyer <mathias@cs.columbia.edu>
#
# Script to eval attack results on PixelDP
#
#
import time
import math
import six
import sys
import json
import os, shutil
from multiprocessing import Pool

import attacks.utils
import datasets

import attacks.params
import models
import models.params
from flags import FLAGS

import tensorflow as tf
import numpy as np
import random
import time

max_batch_size = {
  'madry':           250,
  'pixeldp_resnet':  250,
  'pixeldp_cnn':     1000,
  'inception_model': 160
}
SEED = 1234
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def train_one(dataset, model_class, model_params, attack_class, attack_params,
              dir_name=None, dev='/cpu:0'):

    gpu = int(dev.split(":")[1]) + FLAGS.min_gpu_number
    if gpu >= 16:
        gpu -= 16
    dev = "{}:{}".format(dev.split(":")[0], gpu)

    print("Trainning attack on dev:{}".format(dev), "\n", attack_params)

    with tf.device(dev):
        if dir_name == None:
            dir_name = FLAGS.models_dir

        model_dir  = os.path.join(dir_name, models.params.name_from_params(model_class, model_params))
        attack_dir = os.path.join(model_dir, 'attack_results',
                attacks.params.name_from_params(attack_params))

        if not os.path.exists(attack_dir):
            os.makedirs(attack_dir)

        if dataset == None:
            dataset = FLAGS.dataset

        tot_batch_size   = max_batch_size[models.name_from_module(model_class)]
        # Some book keeping to maximize the GPU usage depending on the attack
        # requirement.
        images_per_batch = attack_class.Attack.image_num_per_batch_train(
                tot_batch_size, attack_params)
        images_per_batch = min(attack_params.num_examples, images_per_batch)
        images, labels = datasets.build_input(
            dataset,
            FLAGS.data_path,
            images_per_batch,
            model_params.image_standardization,
            'eval'
        )
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = str(dev.split(":")[-1])
        sess = tf.Session(config=config)
        tf.train.start_queue_runners(sess)
        coord = tf.train.Coordinator()

        model_params = models.params.update(model_params, 'batch_size', images_per_batch)
        model_params = models.params.update(model_params, 'n_draws',
                attack_params.n_draws_attack)
        if model_class.__name__ == 'madry':
            model = model_class.Model()
        else:
            model = model_class.Model(model_params, None, None, 'eval')

        boxmin = -0.5
        boxmax = 0.5
        binary_search_steps = 9
        learning_rate = 1e-2
        initial_const = 1e-3
        if 'imagenet' in model_dir:
            initial_const /= 10
            learning_rate /= 5
            boxmin = 0.0
            boxmax = 1.0

        attack = attack_class.Attack(sess, model, model_params,
            images.shape, labels.shape,
            attack_params, model_dir, boxmin=boxmin, boxmax=boxmax,
            binary_search_steps=binary_search_steps,
            learning_rate=learning_rate,
            initial_const=initial_const)

        # seek checkpoints
        summary_writer = tf.summary.FileWriter(model_dir)
        try:
            ckpt_state = tf.train.get_checkpoint_state(model_dir)
        except tf.errors.OutOfRangeError as e:
            print('Cannot restore checkpoint: ', e)
            return
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print('\n\n\n\t *** No model to eval yet at: {}\n\n\n'.\
                    format(model_dir))
            return
        print('Loading checkpoint ', ckpt_state.model_checkpoint_path)

        # Figure out variables to restore.
        if 'imagenet' in model_dir and model_params.attack_norm_bound > .0:
            model_vars = [x for x in tf.global_variables()\
                            if "attack" not in x.name\
                            and "global_step" not in x.name\
                            and "encoder" not in x.name\
                            and "decoder" not in x.name]
        else:
            model_vars = [x for x in tf.global_variables() if "attack" not in x.name]

        saver = tf.train.Saver(var_list=model_vars)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord,
                                             daemon=True, start=True))

        num_iter = int(math.ceil(attack_params.num_examples / images_per_batch))
        print("BATCHES: {}".format(num_iter))
        for step in range(1, num_iter + 1):

            if attack.autoencoder is not None:
                imgs, labs = sess.run([images, labels],
                                      {attack.autoencoder.noise_scale:1.0})
            else:
                imgs, labs = sess.run([images, labels])

            print("New attack batch, {} images".format(images_per_batch))
            print("Starting step: {}".format(step))
            start_time = time.time()
            for restart_i in range(1, attack_params.restarts + 1):
                print("Train:: Starting restart {}".format(restart_i))
                # Skip existing batches
                if attacks.utils.check_batch_exitst(attack_dir, step, attack_params,
                                                    restart_i):
                    print("Skipping:", attack_dir, step, restart_i)
                    continue

                # run attack
                adv_img    = attack.run(imgs, labs, restart_i)
                adv_labs = labs
                attacks.utils.save_batch(attack_dir, adv_img, imgs,
                                         adv_labs, labs, step, attack_params,
                                         restart_i)
                print("Seconds per batch:", (int(((time.time() - start_time)/\
                        max(1, attack_params.n_draws_attack)))))

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=2)
        time.sleep(5)

# Params is a list of (model, model_params, attack_params) tupples
def train(params):
    pass

