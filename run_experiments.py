import tensorflow as tf

from experiments import imagenet_eval, cifar100_eval, svhn_eval, mnist_eval, cifar10_eval, cifar10_atk_acc_comp, cifar10_robust_prec, svhn_atk_acc_comp, cifar10_img_noise_eval

def run():
    # svhn_eval.run()
    # cifar10_robust_prec.run()
    # cifar10_img_noise_eval.run()
    # imagenet_eval.run()
    # cifar100_eval.run()
    # attacks_eval_mnist.run()
    # attacks_eval_cifar.run()

    #  cifar10_eval.run(plots_only=True)
    #  cifar10_atk_acc_comp.run(plots_only=True)
    #  cifar10_robust_prec.run(plots_only=True)
    #  svhn_eval.run(plots_only=True)
    #  svhn_atk_acc_comp.run(plots_only=True)
    #  cifar100_eval.run(plots_only=True)
    #  mnist_eval.run(plots_only=True)
    imagenet_eval.run(plots_only=True)

def main(_):
    run()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
