from attacks import carlini, pgd, carlini_robust_precision

def module_from_name(name):
    if name == 'pgd':
        return pgd
    elif name == 'carlini':
        return carlini
    elif name == 'carlini_robust_precision':
        return carlini_robust_precision
    else:
        raise ValueError('Attack "{}" not supported'.format(name))

def name_from_module(module):
    return module.__name__.split('.')[-1]
