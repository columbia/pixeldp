# iterative Fast Gradient Sign Method

import sys
from six.moves import xrange

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


min_lr  = 1.0/256
eps     = 1e-28


class Attack:
    def __init__(self, sess, model, model_params,
            inputs_shape, labels_shape,
            attack_params, boxmin=-0.5, boxmax=0.5):

        image_placeholder = tf.placeholder(tf.float32, inputs_shape)
        image_modification_placeholder = tf.placeholder(tf.float32, inputs_shape)
        label_placeholder = tf.placeholder(tf.int32, labels_shape)
        model_input = image_placeholder + image_modification_placeholder

        model.build_graph(model_input, label_placeholder)

        if attack_params.attack_norm == 'l_inf':
            self.ord = np.inf
        elif attack_params.attack_norm == 'l2':
            self.ord = 2
        elif attack_params.attack_norm == 'l1':
            self.ord = 1
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError('ord must be np.inf, 1, 2.')

        self.min_pix = min_pix
        self.max_pix = max_pix

        self.sess = sess
        self.attack_params = attack_params
        self.budget = attack_params.max_attack_size
        self.learning_rate = min_lr

        if attack_params.attack_norm in ['l2', 'l1']:
            self.learning_rate = 2.5 * self.budget / attack_params.sgd_iterations

        # fix number of iterations
        self.nb_iter = attack_params.sgd_iterations

        self.shape = image_placeholder.shape
        self.noise_scale = 1.0

        self.timg = image_placeholder
        self.modifier = image_modification_placeholder
        self.tlab = label_placeholder


        self.adv_timg = model_input
        self.model = model
        self.output = model.predictions

        self.concated_tlab = self.tlab
        for _ in range(1, self.attack_params.n_draws_attack):
            self.concated_tlab = tf.concat([self.concated_tlab,  self.tlab], 0)

        self.loss  = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.output,
            labels=self.concated_tlab
        )

        # run attack according to: https://arxiv.org/pdf/1611.01236.pdf (page 3)
        #  if self.targeted:
            #  self.loss = -self.loss

        grad, = tf.gradients(self.loss, self.adv_timg)
        self.grad = grad

        reduc_ind = list(xrange(1, len(self.grad.get_shape())))
        if attack_params.attack_norm in ['l2', 'l1']:
            if attack_params.attack_norm == 'l2':
                norm = tf.sqrt(tf.reduce_sum(tf.square(grad),
                                        reduction_indices=reduc_ind,
                                        keep_dims=True))
            elif attack_params.attack_norm == 'l1':
                norm = tf.reduce_sum(tf.abs(grad),
                                     reduction_indices=reduc_ind,
                                     keep_dims=True)
            self.unsafe_norm = norm
            self.safe_norm = tf.maximum(self.unsafe_norm, eps)
            self.normalized_grad = self.grad / self.safe_norm
            self.normalized_scaled_grad = self.learning_rate * self.normalized_grad
        else:                            # apply sign function for l-inf attacks
            norm = tf.reduce_sum(tf.square(grad),
                                        reduction_indices=reduc_ind,
                                        keep_dims=True)
            self.unsafe_norm = norm
            self.safe_norm = tf.maximum(self.unsafe_norm, eps)
            self.normalized_grad = self.grad
            self.normalized_scaled_grad = self.learning_rate *\
                    tf.sign(self.normalized_grad)

        self.adv_timg = tf.add(self.adv_timg, self.normalized_scaled_grad)

        # cap updates
        self.modifier_update = self._clip_update(self.timg, self.adv_timg,
                                                 self.ord, self.budget)

        # collect some debug info
        reduc_ind = list(xrange(1, len(self.timg.get_shape())))
        self.signal = tf.sqrt(tf.reduce_sum(tf.square(self.timg),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True))

        self.noise1 = tf.reduce_sum(tf.abs(self.modifier_update),
                                            reduction_indices=reduc_ind,
                                            keep_dims=True)

        square_sum2 = tf.reduce_sum(tf.square(self.modifier_update),
                                            reduction_indices=reduc_ind,
                                            keep_dims=True)

        self.reverse_noise2 = tf.rsqrt(tf.maximum(square_sum2, eps))
        self.noise2 = 1 / self.reverse_noise2


    def _clip_update(self, input, perturbed_input, ord, budget):
        """ Clipping perturbation modifier to self.ord norm ball """

        #  modifier = perturbed_input - input
        #  1. Remove the the parts that bring us outside the bounds.
        modifier = tf.clip_by_value(perturbed_input, self.min_pix, self.max_pix) - input
        reduc_ind = list(xrange(1, len(modifier.get_shape())))

        if ord not in [np.inf, 1, 2]:
            raise ValueError('ord must be np.inf, 1, , 2.')

        # 2. Normalize
        if ord == np.inf:
            update = tf.clip_by_value(modifier, -budget, budget)
        elif ord == 1:
            norm1 = tf.reduce_sum(tf.abs(modifier),
                                 reduction_indices=reduc_ind,
                                 keep_dims=True)
            norm1 = tf.maximum(norm1, eps)
            final_norm1 = tf.clip_by_value(norm1, eps, budget)
            update = final_norm1 * modifier / norm1
        elif ord == 2:
            square_sum2 = tf.reduce_sum(tf.square(modifier),
                                       reduction_indices=reduc_ind,
                                       keep_dims=True)
            #  reverse_norm2 = tf.rsqrt(tf.maximum(square_sum2, eps))
            norm2 = tf.sqrt(tf.maximum(square_sum2, eps))
            final_norm2 = tf.clip_by_value(norm2, eps, budget)
            update = final_norm2 * modifier / norm2

        # 3. Regardless of the attack norm, clip the update so that the image is
        # still within bounds (should be redundant with 1.).
        return tf.clip_by_value(input + update, self.min_pix, self.max_pix) - input

    def run(self, imgs, labs, restart_i):

        #  # start modifier with a tiny random jump away for the initial point
        #  tf.set_random_seed(self.SEED)

        print(self.attack_params)
        print("Iterations:{}\nStep:{}\nBudget:{}".\
                format(self.nb_iter, self.learning_rate, self.budget))

        # start with a random tiny jump
        # keep the small jump so that cifar10/100 are consistent

        if restart_i == 1:
            modifier = tf.zeros(self.shape)
        elif self.attack_params.attack_norm == 'l2':
            modifier = tf.random_normal(self.shape)
            modifier = 0.5 * self.budget * tf.nn.l2_normalize(modifier)
        else:
            modifier = tf.random_uniform(self.shape, -min_lr, min_lr)
        # modifier = tf.random_normal(self.shape, 0, self.budget)
        modifier = self._clip_update(imgs, imgs + modifier, self.ord,
                                     self.budget)
        modifier = self.sess.run(modifier)

        # set up arguments for first attack iteration
        args = {
            self.timg: imgs, self.tlab: labs,
            self.modifier: modifier,
            self.model.noise_scale: self.noise_scale
        }

        for iteration in range(self.nb_iter):
            modifier, noise2, noise1, safe_norm, unsafe_norm, grad,\
                    normalized_scaled_grad = self.sess.run(
                [
                    self.modifier_update,
                    self.noise2,
                    self.noise1,
                    self.safe_norm,
                    self.unsafe_norm,
                    self.grad,
                    self.normalized_scaled_grad,
                ],
                args
            )
            args[self.modifier] = modifier
            print("{}: modifier (2-norm): {}".format(iteration,
                                                     np.linalg.norm(modifier[0])))
            print("safe norm %.50f" % safe_norm[0])
            print("unsafe norm %.50f" % unsafe_norm[0])
            print("raw gradient ", grad[0][0][0])
            print("normalized scaled gradient ", normalized_scaled_grad[0][0][0])
            print("Iteration: {:d}| L1: {:.3f}| L2: {:.3f} - {}".format(
                iteration, np.mean(noise1), np.mean(noise2), self.budget))

        print("modifier:", modifier[0][0][0])

        # add last perturbation
        adv_images = imgs + modifier

        return np.clip(adv_images, self.min_pix, self.max_pix)

    def image_num_per_batch_train(tot_batch_size, attack_params):
        return max(1, tot_batch_size // attack_params.n_draws_attack)

    def image_num_per_batch_eval(tot_batch_size, attack_params):
        return max(1, tot_batch_size // attack_params.n_draws_eval)

