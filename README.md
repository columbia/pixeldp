Research paper: https://arxiv.org/abs/1802.03471

## Train the model.

Chose the model ll.174-176, and the parameters ll. 111-142 in main.py. See
models/params.py for parameter usage. Then, for instance:

    python3 main.py --num_gpus 1 --dataset cifar10 --mode train

## Eval the model.

Chose parameters in main.py, then:

    python3 main.py --num_gpus 1 --dataset cifar10 --mode eval

The eval data is logged in eval_data.json

## Other "modes"

    attack, attack_eval, plot

For the attack, the attack type and parameters are ll. 152-181 in main.py.

## Files:

The most basic model is in models/pixeldp_cnn.py, with robustness tests in
models/utils/robustness.py, and trained/evaluated through
models/{train,evaluate}.py.

Attacks are implemented in attacks/carlini.py and attacks/pgd.py.

