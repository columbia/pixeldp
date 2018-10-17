"""Template for an experiment..
"""
import time, os, math
from multiprocessing import Pool

import models.params
from datasets import cifar, mnist, svhn
import models
from models.train import train
from models.evaluate import evaluate
from models import pixeldp_cnn, pixeldp_resnet, madry
import attacks
from attacks import pgd, params, train_attack, evaluate_attack
import plots.plot_robust_accuracy
import plots.plot_accuracy_under_attack

import tensorflow as tf

import numpy as np

from flags import FLAGS


def train_eval_model(args):
    dataset_name = args[0]
    model_name = args[1]
    model = models.module_from_name(model_name)
    param_dict = args[2]
    current_gpu = args[3]


    dir_name = os.path.join(FLAGS.models_dir, dataset_name)

    param_dict['batch_size'] = 128
    param_dict['n_draws']    = 1
    hps   = models.params.HParams(**param_dict)


    if model_name == 'madry':
        madry.Model.maybe_download_and_extract(FLAGS.models_dir)
    else:
        print("Running on GPU {}\n\t{}".format(current_gpu, hps))
        with tf.Graph().as_default():
            train(hps, model, dataset=dataset_name, dir_name=dir_name,
                  dev='/gpu:{}'.format(current_gpu))

    compute_robustness = True
    if model_name == 'madry':
        compute_robustness = False
        param_dict['batch_size'] = 2000
        param_dict['n_draws']    = 1
    elif param_dict['noise_after_n_layers'] < 0:
        compute_robustness = False
        param_dict['batch_size'] = 100
        param_dict['n_draws']    = 1
    else:
        param_dict['batch_size'] = 1
        param_dict['n_draws']    = 2000

    hps   = models.params.HParams(**param_dict)
    with tf.Graph().as_default():
        evaluate(hps, model, dataset=dataset_name, dir_name=dir_name,
                compute_robustness=compute_robustness,
                dev='/gpu:{}'.format(current_gpu))
    return hps, model_name

def train_eval_attack(args):
    dataset_name = args[0]
    hps = args[1]
    model_name = args[2]
    model = models.module_from_name(model_name)
    attack_param_dict = args[3]
    current_gpu = args[4]
    dir_name = os.path.join(FLAGS.models_dir, dataset_name)

    attack_params = attacks.params.AttackParams(**attack_param_dict)
    atk = attacks.module_from_name(attack_params.attack_methodolody)

    with tf.Graph().as_default():
        train_attack.train_one(
                dataset_name, model, hps, atk, attack_params, dir_name=dir_name,
                dev='/gpu:{}'.format(current_gpu)
        )

    with tf.Graph().as_default():
        evaluate_attack.evaluate_one(
                dataset_name, model, hps, atk, attack_params, dir_name=dir_name,
                dev='/gpu:{}'.format(current_gpu)
        )

    return hps, attack_params, model_name

