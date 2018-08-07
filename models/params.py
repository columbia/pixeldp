from collections import namedtuple

HParams = namedtuple('HParams',
                     'batch_size, num_classes, lrn_rate, lrn_rte_changes, '
                     'lrn_rte_vals, num_residual_units, use_bottleneck, '
                     'weight_decay_rate, relu_leakiness, optimizer, '
                     'image_standardization, dropout, image_size, '
                     'n_draws, dp_epsilon, dp_delta, attack_norm, '
                     'attack_norm_bound, sensitivity_norm, '
                     'sensitivity_control_scheme, '
                     'noise_after_n_layers, layer_sensitivity_bounds, '
                     'noise_after_activation')

