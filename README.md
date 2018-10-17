## Train the model.

Chose parameters in main.py, then, e.g.:

    python3 main.py --num_gpus 1 --dataset cifar10 --mode train

## Eval the model.

Chose parameters in main.py, then:

    python3 main.py --num_gpus 1 --dataset cifar10 --mode eval

The eval data is logged in eval_data.json

## Other "modes"

    attack, attack_eval, plot

## Files:

The most basic model is in models/pixeldp_cnn.py, with robustness tests in
models/utils/robustness.py, and trained/evaluated through
models/{train,evaluate}.py.

Attacks are implemented in attacks/carlini.py and attacks/pgd.py.

