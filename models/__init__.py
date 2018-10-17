from models import inception_model, pixeldp_cnn, pixeldp_resnet, madry

def module_from_name(name):
    if name == 'pixeldp_cnn':
        return pixeldp_cnn
    elif name == 'pixeldp_resnet':
        return pixeldp_resnet
    elif name == 'madry':
        return madry
    elif name == 'inception-v3':
        return inception_model
    else:
        raise ValueError('Model "{}" not supported'.format(name))

def name_from_module(module):
    return module.__name__.split('.')[-1]
