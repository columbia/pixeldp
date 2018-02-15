# Setup

Download the CIFAR dataset and setup the right path in main.py (or as
command line args).

# Train the model.

Chose parameters in main.py ll 289-301

    python3 main.py --num_gpus 1

# Eval the model.

Chose parameters in main.py ll 289-301

    python3 main.py --num_gpus 1 --mode eval

The eval data is logged in eval_data.json
