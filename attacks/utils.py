import json
import os, sys

import numpy as np


def check_batch_exitst(attack_dir, batch_no, params, restart):
    attack_dir = attack_dir
    if not os.path.exists(attack_dir):
        return False

    path = attack_dir + "/" + "batch-{}.npy".format(batch_no)
    if not os.path.exists(path):
        return False

    path = attack_dir + "/" + "advbatch-{}-r-{}.npy".format(batch_no, restart)
    if not os.path.exists(path):
        return False

    path = attack_dir + "/" + "labs-{}.npy".format(batch_no)
    if not os.path.exists(path):
        return False

    path = attack_dir + "/" + "advlabs-{}.npy".format(batch_no)
    if not os.path.exists(path):
        return False

    return True


def save_batch(attack_dir, adv_inputs, inputs, adv_labs, labs, batch_no, params,
               restart):
    """Helper for saving adversarial data (will be loaded by eval_attack)
    """
    attack_dir = attack_dir
    if not os.path.exists(attack_dir):
        os.makedirs(attack_dir)
    inputs = np.array(inputs)

    path = attack_dir + "/" + "batch-{}".format(batch_no)
    np.save(path, inputs)

    adv_inputs = np.array(adv_inputs)
    path = attack_dir + "/" + "advbatch-{}-r-{}".format(batch_no, restart)
    np.save(path, adv_inputs)
    #labs = np.argmax(labs, axis=1)
    path = attack_dir + "/" + "labs-{}".format(batch_no)
    np.save(path, labs)

    #adv_labs = np.argmax(adv_labs, axis=1)
    path = attack_dir + "/" + "advlabs-{}".format(batch_no)
    np.save(path, adv_labs)

    path = attack_dir + "/" + "params.json"
    with open(path, "w") as f:
        json.dump(params, f)


def load_batch(attack_dir, batch_no, restart):
    """Helper to load a batch for evaluation
    """
    if not os.path.exists(attack_dir):
        print("Malformed data directory hierarchy")
        print(attack_dir)
        sys.exit(-1)
        return [], [], [], []

    path = attack_dir + "/" + "batch-{}".format(batch_no)
    inputs = np.load(path + ".npy")

    path = attack_dir + "/" + "advbatch-{}-r-{}".format(batch_no, restart)
    adv_inputs = np.load(path + ".npy")
    path = attack_dir + "/" + "labs-{}".format(batch_no)
    labs = np.load(path + ".npy")

    path = attack_dir + "/" + "advlabs-{}".format(batch_no)
    adv_labs = np.load(path + ".npy")
    return inputs, adv_inputs, labs, adv_labs

