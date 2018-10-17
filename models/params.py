from collections import namedtuple

"""Defines the parameters name tupple to pass to models.

Args:
  name_prefix: prefix prepended to the model's dir name.
  batch_size: minibatch size for training.
  num_classes: Num classes in classification task.
  image_size: Images are image_size x image_size pixels.
  lrn_rate: Initial learning rate for training.
  lrn_rte_changes: Training steps when to change the learning rate.
  lrn_rte_vals: Learning rate value to change to at corresponding
                lrn_rte_changes step.
  num_residual_units: For ResNet.
  use_bottleneck: For ResNet.
  weight_decay_rate: Weight decay coef if model has a weight decay.
  relu_leakiness: Make relu leaky or not.
  optimizer: Use {sgd,mom} to train.
  image_standardization: Use image_standardization to preprocess inputs.
  n_draws: For PixelDP, number of draws to use (can change between training
           and eval).
  dp_epsilon: DP epsilon param to use.
  dp_delta: DP delta param to use.
  attack_norm_bound: Max attack norm to defend against when computing DP noise.
  attack_norm: The attack norm to use.
  sensitivity_norm: The norm to measure sensitivity in.
  sensitivity_control_scheme: {bound,optimize} sensitivity during training.
  noise_after_n_layers: How many pre-noise layers.
  layer_sensitivity_bounds: How to contro the sensitivity of each pre-noise
                            layer.
  noise_after_activation: Add noise layer before or after the activation fn.
  parseval_loops: How many loops of Parseval projection between each SGD step.
  parseval_step: Parseval projection step size.
  steps_num: Number of training steps.
  eval_data_size: Size of the eval dataset.
"""

HParams = namedtuple('HParams',
                     'name_prefix, batch_size, num_classes, lrn_rate, '
                     'lrn_rte_changes, lrn_rte_vals, num_residual_units, '
                     'use_bottleneck, weight_decay_rate, relu_leakiness, '
                     'optimizer, image_standardization, image_size, n_channels,'
                     'n_draws, dp_epsilon, dp_delta, attack_norm, '
                     'robustness_confidence_proba, '
                     'attack_norm_bound, sensitivity_norm, '
                     'sensitivity_control_scheme, '
                     'noise_after_n_layers, layer_sensitivity_bounds, '
                     'noise_after_activation, parseval_loops, parseval_step, '
                     'steps_num, eval_data_size')

def update(params, key, value):
    params = params._asdict()
    params[key] = value
    return HParams(**params)

def name_from_params(model, hps):
    if type(model) == str and model == 'robustop':
        n = 'robustop'
    elif 'madry' in model.__name__:
        n = 'madry'
    else:
        n = ("{}_attack_norm_{}_size_{}_{}_prenoise_layers_"
             "sensitivity_{}_scheme_{}_activation_{}").format(
            model.__name__.split('.')[-1],
            hps.attack_norm,
            hps.attack_norm_bound,
            hps.noise_after_n_layers,
            hps.sensitivity_norm,
            hps.sensitivity_control_scheme,
            'postnoise' if hps.noise_after_activation else 'prenoise'
        )

    if len(hps.name_prefix) > 0:
        n = "{}_{}".format(hps.name_prefix, n)

    return n

