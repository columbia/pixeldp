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
    'pixeldp_cnn':      5000,
    'inception_model':  150
}
def evaluate_one(dataset, model_class, model_params, attack_class,
                 attack_params, dir_name=None, compute_robustness=True,
                 dev='/cpu:0'):

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

    assert(attack_params.restarts == 1)

    tot_batch_size_atk = train_attack.max_batch_size[models.name_from_module(model_class)]
    images_per_batch_attack = min(
            attack_params.num_examples,
            attack_class.Attack.image_num_per_batch_train(
                tot_batch_size_atk, attack_params))

    print(attack_dir)

    xs = [round(x, 2) for x in np.arange(0, 5.0, 0.1)]
    robust_true  = [0] * len(xs)
    robust_false = [0] * len(xs)
    tot          = 0

    num_iter = int(math.ceil(attack_params.num_examples / images_per_batch_attack))
    for step in range(0, num_iter):
        inputs, adv_inputs, labs, adv_labs = attacks.utils.load_batch(
                attack_dir, step + 1, 1)

        print(adv_inputs)
        for l2s in adv_inputs:
            tot += 1
            for i, x in enumerate(xs):
                if min(l2s[1:]) > x:
                    robust_true[i] += 1
                if l2s[-1] <= x:
                    robust_false[i] += 1

    print(list(xs))
    print([x/tot for x in robust_true])
    print([x/tot for x in robust_false])

    data = {
        'x': list(xs),
        'robust': [sum(x) for x in zip(robust_false, robust_true)],
        'robust_false': robust_false,
        'robust_true':  robust_true,
        'tot': tot,
    }
    # Log eval data
    with open(result_path, 'w') as f:
        f.write(json.dumps(data))

    return data

