from collections import namedtuple

HParams = namedtuple('HParams',
                     'batch_size, num_classes, lrn_rate, lrn_rte_changes, lrn_rte_vals, '
                     'num_residual_units, use_bottleneck, '
                     'weight_decay_rate, relu_leakiness, optimizer, '
                     'image_standardization, dropout, image_size, '
                     'n_draws, noise_scheme, attack_norm_bound')

