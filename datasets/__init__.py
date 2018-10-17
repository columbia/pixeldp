def build_input(dataset, data_path, batch_size, standardize_images, mode):
    if dataset == 'mnist':
        from datasets import mnist
        return mnist.build_input(data_path, batch_size, standardize_images, mode)
    elif dataset == 'svhn':
        from datasets import svhn
        return svhn.build_input(data_path, batch_size, standardize_images, mode)
    elif dataset == 'cifar10':
        from datasets import cifar
        return cifar.build_input(dataset, data_path, batch_size, standardize_images, mode)
    elif dataset == 'cifar100':
        from datasets import cifar
        return cifar.build_input(dataset, data_path, batch_size, standardize_images, mode)
    elif dataset == 'imagenet':
        from inception import image_processing
        from inception.imagenet_data import ImagenetData
        images, labels = image_processing.inputs(ImagenetData('validation'),
                                                 batch_size=batch_size)
        import tensorflow as tf
        labels = tf.one_hot(labels, 1001)
        return images, labels
    else:
        raise ValueError("Dataset {} not supported".format(dataset))
