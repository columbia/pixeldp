# Copyright (C) 2018 Vaggelis Atlidakis <vatlidak@cs.columbia.edu> et
# Mathias Lecuyer <mathias@cs.columbia.edu>
#
# Script to eval attack results on PixelDP
#
import tensorflow as tf
import numpy as np
import math
import json
import os, sys, time
from multiprocessing import Pool

import models
import models.params
import attacks.utils
import datasets
from models.utils import robustness

import attacks.params
from attacks import train_attack
from flags import FLAGS

max_batch_size = {
    'madry':            250,
    'pixeldp_resnet':   250,
    'pixeldp_cnn':      1000,
    'inception_model':  160
}
def evaluate_one(dataset, model_class, model_params, attack_class,
                 attack_params, dir_name=None, compute_robustness=True,
                 dev='/cpu:0'):

    gpu = int(dev.split(":")[1]) + FLAGS.min_gpu_number
    gpu = gpu % 16  # for 16 GPUs exps
    dev = "{}:{}".format(dev.split(":")[0], gpu)

    print("Evaluating attack on dev:{}".format(dev), "\n", attack_params)
    with tf.device(dev):
        if dir_name == None:
            dir_name = FLAGS.models_dir

        model_dir  = os.path.join(dir_name, models.params.name_from_params(model_class, model_params))
        attack_dir = os.path.join(model_dir, 'attack_results',
                attacks.params.name_from_params(attack_params))

        # if results are in place, don't redo
        result_path = os.path.join(attack_dir, "eval_data.json")
        if os.path.exists(result_path):
            print("Path: {} exists -- skipping!!!".format(result_path))
            return

        if dataset == None:
            dataset = FLAGS.dataset

        tot_batch_size_atk = train_attack.max_batch_size[models.name_from_module(model_class)]
        tot_batch_size     = max_batch_size[models.name_from_module(model_class)]
        # Some book keeping to maximize the GPU usage depending on the attack
        # requirement.
        images_per_batch_attack = min(
                attack_params.num_examples,
                attack_class.Attack.image_num_per_batch_train(
                    tot_batch_size_atk, attack_params))
        images_per_batch_eval   = min(
                attack_params.num_examples,
                attack_class.Attack.image_num_per_batch_eval(
                    tot_batch_size, attack_params))
        batch_size = min(images_per_batch_attack, images_per_batch_eval)

        image_placeholder = tf.placeholder(tf.float32,
                [batch_size, model_params.image_size,
                 model_params.image_size, model_params.n_channels])
        label_placeholder = tf.placeholder(tf.int32,
                [batch_size, model_params.num_classes])

        model_params = models.params.update(model_params, 'batch_size',
                batch_size)
        model_params = models.params.update(model_params, 'n_draws',
                attack_params.n_draws_eval)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = str(dev.split(":")[-1])
        sess = tf.Session(config=config)

        # Special treatment of imagenet: load inception + autoencoder
        if 'imagenet' in model_dir and model_params.attack_norm_bound > 0.0:
            autoencoder_dir_name = os.path.join(model_dir, "autoencoder_l2_l2_s1_{}_32_32_64_10_8_5_srd1221_srd1221_srd1221".format(model_params.attack_norm_bound))
            autoencoder_params = json.load(
                open(os.path.join(autoencoder_dir_name, "params.json"), "r")
            )
            autoencoder_params['n_draws'] = attack_params.n_draws_eval
            # hyperparams for autoencoder
            autoencoder_hps = tf.contrib.training.HParams()
            for k in autoencoder_params:
                autoencoder_hps.add_hparam(k, autoencoder_params[k])
            autoencoder_hps.batch_size = batch_size*attack_params.n_draws_attack
            autoencoder_hps.autoencoder_dir_name = autoencoder_dir_name
            from models import autoencoder_model
            autoencoder_model = autoencoder_model.Autoencoder(autoencoder_hps,
                                                              image_placeholder,
                                                              image_placeholder,
                                                              "eval")
            autoencoder_model.build_graph()
            autoencoder_variables = []
            for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                autoencoder_variables.append(k)
            autoencoder_saver = tf.train.Saver(autoencoder_variables)
            autoencoder_summary_writer = tf.summary.FileWriter(autoencoder_dir_name)
            try:
                autoencoder_ckpt_state = tf.train.get_checkpoint_state(autoencoder_dir_name)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
            print('Autoencoder: Loading checkpoint',
                            autoencoder_ckpt_state.model_checkpoint_path)
            autoencoder_saver.restore(sess,
                                      autoencoder_ckpt_state.model_checkpoint_path)
            # imagenet dataset loader returns images in [0, 1]
            images = 2*(autoencoder_model.output - 0.5)
            model = model_class.Model(model_params, images, label_placeholder,
                                      'eval')
            model.build_graph()
            inception_variables = []
            for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if k in autoencoder_variables and k.name != "global_step":
                    continue
                if k.name.startswith("DW-encoder") or k.name.startswith("b-encoder")\
                     or k.name.startswith("b-decoder"):
                    continue
                inception_variables.append(k)
            saver = tf.train.Saver(inception_variables)
        else:
            model = model_class.Model(model_params, image_placeholder,
                    label_placeholder, 'eval')
            model.build_graph()
            saver = tf.train.Saver()

        with sess:
            tf.train.start_queue_runners(sess)
            coord = tf.train.Coordinator()

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
            saver.restore(sess, ckpt_state.model_checkpoint_path)

            ops = model.predictions  # results of the softmax layer

            clean_preds = []
            defense_preds = []
            counters = []

            if model.noise_scale != None:
                args = {model.noise_scale: 1.0}
            else:
                args = {}
            if 'imagenet' in model_dir and model_params.attack_norm_bound > 0.0:
                args = {autoencoder_model.noise_scale: 1.0}

            data = {
                'argmax_sum': [],
                'softmax_sum': [],
                'softmax_sqr_sum': [],
                'robustness_from_argmax': [],
                'robustness_from_softmax': [],
                'adv_argmax_sum': [],
                'adv_softmax_sum': [],
                'adv_softmax_sqr_sum': [],
                'adv_robustness_from_argmax': [],
                'adv_robustness_from_softmax': [],
                'pred_truth':  [],
                'adversarial_norm': [],
            }

            num_iter = int(math.ceil(attack_params.num_examples / images_per_batch_attack))
            intra_batch_num_iter = int(math.ceil(images_per_batch_attack / batch_size))
            for step in range(0, num_iter):
                print("Evaluate:: Starting step {}/{}".format(step+1, num_iter))
                pred_truth  = np.zeros([images_per_batch_attack], dtype=int)

                predictions = np.zeros([images_per_batch_attack], dtype=int)

                prediction_votes = np.zeros(
                        [images_per_batch_attack, model_params.num_classes])
                prediction_softmax_sum = np.zeros(
                        [images_per_batch_attack, model_params.num_classes])
                prediction_softmax_sum_sqr = np.zeros(
                        [images_per_batch_attack, model_params.num_classes])

                adv_prediction_votes = np.zeros(
                        [images_per_batch_attack, attack_params.restarts,
                         model_params.num_classes])
                adv_prediction_softmax_sum = np.zeros(
                        [images_per_batch_attack, attack_params.restarts,
                         model_params.num_classes])
                adv_prediction_softmax_sum_sqr = np.zeros(
                        [images_per_batch_attack, attack_params.restarts,
                         model_params.num_classes])

                adv_norm = np.zeros(
                        [images_per_batch_attack, attack_params.restarts])

                for restart in range(0, attack_params.restarts):
                    print("Evaluate:: Starting restart {}/{}".format(
                        restart+1, attack_params.restarts))
                    # Naming is advbatch-1-r-1, advbatch-2-r-1, advbatch-1-r-2 ...
                    inputs, adv_inputs, labs, adv_labs = attacks.utils.load_batch(
                        attack_dir, step + 1, restart + 1)

                    if attack_params.attack_norm == 'l2':
                        norm_ord = 2
                    elif attack_params.attack_norm == 'l_inf':
                        norm_ord = np.inf
                    else:
                        raise ValueError("Attack norm not supported")

                    s = inputs.shape
                    adv_norm_restart = np.linalg.norm(
                            np.reshape(inputs, (s[0], -1)) -  \
                                    np.reshape(adv_inputs, (s[0], -1)),
                            ord=norm_ord,
                            axis=1
                    )
                    adv_norm[:,restart] = adv_norm_restart

                    for intra_batch_step in range(0, intra_batch_num_iter):
                        batch_i_start = intra_batch_step       * batch_size
                        batch_i_end   = min((intra_batch_step + 1) * batch_size,
                                images_per_batch_attack)

                        image_batch     = inputs[batch_i_start:batch_i_end]
                        adv_image_batch = adv_inputs[batch_i_start:batch_i_end]
                        label_batch     = labs[batch_i_start:batch_i_end]

                        # Handle end of batch with wrong size
                        true_batch_size = image_batch.shape[0]
                        if true_batch_size < batch_size:
                            pad_size = batch_size - true_batch_size
                            image_batch = np.pad(image_batch,
                                    [(0, pad_size), (0,0), (0,0), (0,0)],
                                    'constant')
                            adv_image_batch = np.pad(adv_image_batch,
                                    [(0, pad_size), (0,0), (0,0), (0,0)],
                                    'constant')
                            label_batch = np.pad(label_batch,
                                    [(0, pad_size), (0,0)],
                                    'constant')

                        # Predictions on the original image: only on one restart
                        if restart == 0:
                            args[image_placeholder] = image_batch
                            args[label_placeholder] = label_batch

                            softmax = sess.run(ops, args)
                            max_softmax = np.argmax(softmax, axis=1)
                            for i in range(attack_params.n_draws_eval):
                                for j in range(true_batch_size):
                                    abs_j = batch_i_start + j

                                    pred_truth[abs_j] = np.argmax(label_batch[j])

                                    rel_i = i*batch_size+j
                                    pred = max_softmax[rel_i]
                                    prediction_votes[abs_j, pred] += 1
                                    prediction_softmax_sum[abs_j] +=  \
                                        softmax[rel_i]
                                    prediction_softmax_sum_sqr[abs_j] +=  \
                                        np.square(softmax[rel_i])

                        # Predictions on the adversarial image for current
                        # restart
                        args[image_placeholder] = adv_image_batch
                        args[label_placeholder] = label_batch

                        softmax = sess.run(ops, args)
                        max_softmax = np.argmax(softmax, axis=1)
                        for i in range(attack_params.n_draws_eval):
                            for j in range(true_batch_size):
                                abs_j = batch_i_start + j
                                rel_i = i*batch_size+j
                                pred = max_softmax[rel_i]
                                adv_prediction_votes[abs_j, restart, pred] += 1
                                adv_prediction_softmax_sum[abs_j, restart] +=  \
                                        softmax[rel_i]
                                adv_prediction_softmax_sum_sqr[abs_j, restart] \
                                        += np.square(softmax[rel_i])

                predictions = np.argmax(prediction_votes, axis=1)
                adv_predictions = np.argmax(adv_prediction_votes, axis=2)

                data['pred_truth'] += pred_truth.tolist()

                data['adversarial_norm'] += adv_norm.tolist()

                data['argmax_sum'] += prediction_votes.tolist()
                data['softmax_sum'] += prediction_softmax_sum.tolist()
                data['softmax_sqr_sum'] += prediction_softmax_sum_sqr.tolist()

                data['adv_argmax_sum'] += adv_prediction_votes.tolist()
                data['adv_softmax_sum'] += adv_prediction_softmax_sum.tolist()
                data['adv_softmax_sqr_sum'] += adv_prediction_softmax_sum_sqr.tolist()

        sensitivity_multiplier = 1.0
        try:
            with open(model_dir + "/sensitivity_multiplier.json") as f:
                sensitivity_multiplier = float(json.loads(f.read())[0])
        except Exception:
            print("Missing Mulltiplier")
            pass

        # Compute robustness and add it to the eval data.
        if compute_robustness:  # This is used mostly to avoid errors on non pixeldp DNNs
            dp_mechs = {
                'l2': 'gaussian',
                'l1': 'laplace',
            }
            robustness_from_argmax = [robustness.robustness_size_argmax(
                counts=x,
                eta=model_params.robustness_confidence_proba,
                dp_attack_size=model_params.attack_norm_bound,
                dp_epsilon=model_params.dp_epsilon,
                dp_delta=model_params.dp_delta,
                dp_mechanism=dp_mechs[model_params.sensitivity_norm]
                ) / sensitivity_multiplier for x in data['argmax_sum']]
            data['robustness_from_argmax'] = robustness_from_argmax
            robustness_form_softmax = [robustness.robustness_size_softmax(
                tot_sum=data['softmax_sum'][i],
                sqr_sum=data['softmax_sqr_sum'][i],
                counts=data['argmax_sum'][i],
                eta=model_params.robustness_confidence_proba,
                dp_attack_size=model_params.attack_norm_bound,
                dp_epsilon=model_params.dp_epsilon,
                dp_delta=model_params.dp_delta,
                dp_mechanism=dp_mechs[model_params.sensitivity_norm]
                ) / sensitivity_multiplier for i in range(len(data['argmax_sum']))]
            data['robustness_form_softmax'] = robustness_form_softmax
            adv_robustness_from_argmax = [
                [robustness.robustness_size_argmax(
                    counts=x[r],
                    eta=model_params.robustness_confidence_proba,
                    dp_attack_size=model_params.attack_norm_bound,
                    dp_epsilon=model_params.dp_epsilon,
                    dp_delta=model_params.dp_delta,
                    dp_mechanism=dp_mechs[model_params.sensitivity_norm]
                    ) / sensitivity_multiplier for r in range(0, attack_params.restarts)]
                for x in data['adv_argmax_sum']]
            data['adv_robustness_from_argmax'] = adv_robustness_from_argmax
            adv_robustness_form_softmax = [
                [robustness.robustness_size_softmax(
                    tot_sum=data['adv_softmax_sum'][i][r],
                    sqr_sum=data['adv_softmax_sqr_sum'][i][r],
                    counts=data['adv_argmax_sum'][i][r],
                    eta=model_params.robustness_confidence_proba,
                    dp_attack_size=model_params.attack_norm_bound,
                    dp_epsilon=model_params.dp_epsilon,
                    dp_delta=model_params.dp_delta,
                    dp_mechanism=dp_mechs[model_params.sensitivity_norm]
                    ) / sensitivity_multiplier for r in range(0, attack_params.restarts)]
                for i in range(len(data['adv_argmax_sum']))]
            data['adv_robustness_form_softmax'] = adv_robustness_form_softmax

        data['sensitivity_mult_used'] = sensitivity_multiplier

        # Log eval data
        with open(result_path, 'w') as f:
            f.write(json.dumps(data))

        return data

