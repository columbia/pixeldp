from collections import namedtuple

"""Defines the attack parameters name tupple.

Args:
  restarts: how many restarts with small random initial perturbation to perform.
  n_draws_attack: number of pixeldp noise draws at each attack gradient step.
  n_draws_eval: number of pixeldp noise draws to evaluate if an attack is
                successful.
  attack_norm: Measure the attack in {l_inf,l_1,l_2} norms.
  max_attack_size: Max attack size as measured with attack_norm.
  num_examples: Number of examples to attack.
  attack_methodolody: Which attack to use in {pgd,c_w}.
  targeted: Bool, is the attack targeted or untargeted?
  sgd_iterations: Max number of gradient steps to take.
"""

AttackParams = namedtuple('AttackParams',
                          'restarts, n_draws_attack, n_draws_eval, attack_norm, '
                          'max_attack_size, num_examples, '
                          'attack_methodolody, targeted, sgd_iterations, use_softmax')
AttackParamsPrec = namedtuple('AttackParamsPrec',
                          'restarts, n_draws_attack, n_draws_eval, attack_norm, '
                          'max_attack_size, num_examples, '
                          'attack_methodolody, targeted, sgd_iterations, use_softmax, T')

def name_from_params(attack_params):
    params = attack_params._asdict()
    if 'T' in params:
        return ('{attack_methodolody}_attack_norm_{attack_norm}_'
                'size_{max_attack_size}_restarts_{restarts}_'
                'targeted_{targeted}_softmax_{use_softmax}_T_{T}'
        ).format(**params)
    return ('{attack_methodolody}_attack_norm_{attack_norm}_'
            'size_{max_attack_size}_restarts_{restarts}_'
            'targeted_{targeted}_softmax_{use_softmax}').format(**params)

def update(params, key, value):
    params = params._asdict()
    params[key] = value
    return AttackParams(**params)
