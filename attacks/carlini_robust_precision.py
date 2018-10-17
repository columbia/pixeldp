## Copyright (C) 2018, Mathias Lecuyer <mathias.lecuyer@gmail.com>.

## Adapted from: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
## by Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file https://github.com/carlini/nn_robust_attacks/blob/master/LICENSE.

#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
        #  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
from six.moves import xrange

import tensorflow as tf
import numpy as np
import time, math

import models
from models.utils import robustness

FLAGS = tf.app.flags.FLAGS


BINARY_SEARCH_STEPS = 9   # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000     # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2      # larger values converge faster to less accurate results
TARGETED = False          # should we target one specific class? or just be wrong?
CONFIDENCE = 0            # how strong the adversarial example should be
INITIAL_CONST = 1e-3      # the initial constant c to pick as a first guess

class Attack:
    def __init__(self, sess, model, model_params,
            inputs_shape, labels_shape,
            attack_params,
            batch_size=1,
            confidence = CONFIDENCE,
            targeted = TARGETED,
            binary_search_steps = BINARY_SEARCH_STEPS,
            max_iterations = MAX_ITERATIONS,
            abort_early = ABORT_EARLY,
            initial_const = INITIAL_CONST,
            boxmin = -0.5, boxmax = 0.5,
            learning_rate = LEARNING_RATE
            ):
        """
        The L_2 optimized attack.
        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.
        Returns adversarial examples for the supplied model.
        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        self.sess = sess
        self.TARGETED = targeted
        self.MAX_ITERATIONS = attack_params.sgd_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.T = attack_params.T
        self.sensitivity_multiplier = None
        self.autoencoder = None

        self.model_params = model_params
        batch_size = inputs_shape[0].value
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        if attack_params.attack_norm != 'l2':
            raise ValueError('This is an L2 attack.')

        self.attack_params = attack_params
        self.budget = attack_params.max_attack_size
        self.LEARNING_RATE = learning_rate

        self.noise_scale = 1.0
        self.model = model
        shape = inputs_shape
        self.shape = shape

        with tf.variable_scope('attack'):
            # the variable we're going to optimize over
            modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

            # these are variables to be more efficient in sending data to tf
            self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
            self.tlab = tf.Variable(np.zeros(labels_shape), dtype=tf.float32)
            self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg  = tf.placeholder(tf.float32, shape)
        self.assign_tlab  = tf.placeholder(tf.float32, labels_shape)
        self.assign_const = tf.placeholder(tf.float32, [batch_size])

        # for random restarts
        self.assign_modifier = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul  = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg  = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus

        self.concated_tlab = self.tlab
        self.concated_const = self.const
        for _ in range(1, self.attack_params.n_draws_attack):
            self.concated_tlab  = tf.concat([self.concated_tlab,  self.tlab], 0)
            self.concated_const = tf.concat([self.concated_const,  self.const], 0)
        # prediction BEFORE-SOFTMAX of the model
        model.build_graph(self.newimg, self.tlab)
        self.output = model.logits
        self.predictions = model.predictions

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.concated_tlab)*self.output,1)
        other = tf.reduce_max((1-self.concated_tlab)*self.output - (self.concated_tlab*10000),1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            #  loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)
            loss1 = real - other

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        #  self.loss2_replicated = self.attack_params.n_draws_attack * self.loss2
        self.loss2_replicated = self.loss2
        self.loss1 = tf.reduce_sum(self.concated_const*loss1)
        self.loss = self.loss1 + self.loss2_replicated

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        with tf.variable_scope('attack'):
            optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
            self.train = optimizer.minimize(self.loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(modifier.assign(self.assign_modifier))

        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def run(self, imgs, targets, restart_i):
        """
        Perform the L_2 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size], restart_i))
        return np.array(r)

    def attack_batch(self, imgs, labs, restart_i):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y,is_robust):
            if not is_robust:
                # An attack is successful only when it give a wrong robust
                # prediction.
                return False

            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        dp_mechs = {
            'l2': 'gaussian',
            'l1': 'laplace',
        }
        if self.sensitivity_multiplier == None:
            self.sensitivity_multiplier = self.sess.run(self.model.pre_noise_sensitivity())

        batch_size = self.batch_size

        # convert to tanh-space
        _imgs = imgs
        imgs = np.arctanh((_imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size

        o_true_robust_min_l2      = [1e10]*batch_size
        o_true_not_robust_min_l2  = [1e10]*batch_size
        o_false_not_robust_min_l2 = [1e10]*batch_size
        o_false_robust_min_l2     = [1e10]*batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print("Starting step {}".format(outer_step))
            print(o_bestl2)
            o_final = zip(o_true_robust_min_l2,
                          o_true_not_robust_min_l2,
                          o_false_not_robust_min_l2,
                          o_false_robust_min_l2)
            print(np.array(list(o_final)))
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            if restart_i == 1:
                # First restart, modifier is 0.
                random_start = tf.zeros(self.shape)
                random_start = self.sess.run(random_start)
            else:
                # init the modifier with a random perturbation of the original
                # image.
                random_start = tf.random_normal(self.shape)
                random_start = 0.2 * tf.nn.l2_normalize(random_start)
                random_start = self.sess.run(random_start)
                random_start = np.clip(_imgs + random_start, -.5, .5)
                random_start = np.arctanh((random_start - self.boxplus) / self.boxmul * 0.999999) - imgs

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST,
                                       self.assign_modifier: random_start})

            prev = 1e6
            if self.model.noise_scale != None:
                args = { self.model.noise_scale: self.noise_scale }
            else:
                args = {}

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l1, l2, l2s, scores, nimg, softmax_predictions = self.sess.run(
                        [self.train,
                         self.loss,
                         self.loss1,
                         self.loss2,
                         self.l2dist,
                         self.output,
                         self.newimg,
                         self.predictions],
                        args
                    )

                print("Loop {}".format(iteration), (l, l1, l2))

                if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                        raise Exception("The output of model.predict should ",
                            "return the pre-softmax layer. It looks like you ",
                            "are returning the probability vector ",
                            "(post-softmax).")

                if iteration % 10 == 0:
                    # Make many predictions to determinie if the attacks is a
                    # success.
                    n_runs = 0
                    if self.attack_params.use_softmax:
                        predictions_form_softmax = np.zeros(
                            [self.batch_size, self.model_params.num_classes]
                        )
                        predictions_form_softmax_squared = np.zeros(
                            [self.batch_size, self.model_params.num_classes]
                        )
                        predictions_form_argmax  = np.zeros(
                            [self.batch_size, self.model_params.num_classes]
                        )
                    else:
                        predictions_form_argmax  = np.zeros(
                            [self.batch_size, self.model_params.num_classes]
                        )

                    argmax_predictions = np.argmax(softmax_predictions, axis=1)
                    while True:
                        for i in range(self.attack_params.n_draws_attack):
                            n_runs += 1
                            for j in range(self.batch_size):
                                _i = i * batch_size + j
                                pred = argmax_predictions[_i]
                                if self.attack_params.use_softmax:
                                    predictions_form_softmax[j] += softmax_predictions[_i]
                                    predictions_form_softmax_squared[j] += np.square(softmax_predictions[_i])
                                    predictions_form_argmax[j, pred] += 1
                                else:
                                    predictions_form_argmax[j, pred] += 1

                        if self.attack_params.n_draws_eval <= n_runs:
                            break
                        else:
                            predictions = self.sess.run(self.predictions, args)

                    if self.attack_params.use_softmax:
                        final_predictions = predictions_form_softmax
                        is_correct = []
                        is_robust  = []
                        for i in range(self.batch_size):
                            is_correct.append(
                                np.argmax(batchlab[i]) == np.argmax(final_predictions[i])
                            )
                            robustness_form_softmax = robustness.robustness_size_softmax(
                                tot_sum=predictions_form_softmax[i],
                                sqr_sum=predictions_form_softmax_squared[i],
                                counts=predictions_form_argmax[i],
                                eta=self.model_params.robustness_confidence_proba,
                                dp_attack_size=self.model_params.attack_norm_bound,
                                dp_epsilon=self.model_params.dp_epsilon,
                                dp_delta=self.model_params.dp_delta,
                                dp_mechanism=dp_mechs[self.model_params.sensitivity_norm]
                                ) / self.sensitivity_multiplier
                            is_robust.append(robustness_form_softmax >= self.T)
                    else:
                        final_predictions = predictions_form_argmax
                        is_correct = []
                        is_robust = []
                        for i in range(self.batch_size):
                            is_correct.append(
                                np.argmax(batchlab[i]) == np.argmax(final_predictions[i])
                            )
                            robustness_from_argmax = robustness.robustness_size_argmax(
                                counts=predictions_form_argmax[i],
                                eta=self.model_params.robustness_confidence_proba,
                                dp_attack_size=self.model_params.attack_norm_bound,
                                dp_epsilon=self.model_params.dp_epsilon,
                                dp_delta=self.model_params.dp_delta,
                                dp_mechanism=dp_mechs[self.model_params.sensitivity_norm]
                                ) / self.sensitivity_multiplier
                            is_robust.append(robustness_from_argmax >= self.T)

                    # adjust the best result found so far
                    for e,(l2,sc,ii) in enumerate(zip(l2s,final_predictions,nimg)):
                        l2 = math.sqrt(l2)
                        if l2 < bestl2[e] and not is_correct[e] and is_robust[e]:
                            bestl2[e] = l2
                            bestscore[e] = np.argmax(sc)
                        if l2 < o_bestl2[e] and not is_correct[e] and is_robust[e]:
                            o_bestl2[e] = l2
                            o_bestscore[e] = np.argmax(sc)
                            o_bestattack[e] = ii
                        if l2 < o_true_robust_min_l2[e] and  \
                                is_correct[e] and is_robust[e]:
                            o_true_robust_min_l2[e] = l2
                        if l2 < o_true_not_robust_min_l2[e] and  \
                                is_correct[e] and not is_robust[e]:
                            o_true_not_robust_min_l2[e] = l2
                        if l2 < o_false_not_robust_min_l2[e] and  \
                                not is_correct[e] and not is_robust[e]:
                            o_false_not_robust_min_l2[e] = l2
                        if l2 < o_false_robust_min_l2[e] and  \
                                not is_correct[e] and is_robust[e]:
                            o_false_robust_min_l2[e] = l2

                    # check if we should abort search if we're getting nowhere.
                    if self.ABORT_EARLY and iteration % 20 == 0:
                        if l > prev*.9999:
                            break
                        prev = l

            # adjust the constant as needed
            for e in range(batch_size):
                if bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        o_final = zip(o_true_robust_min_l2,
                      o_true_not_robust_min_l2,
                      o_false_not_robust_min_l2,
                      o_false_robust_min_l2)
        o_final = list(o_final)
        return np.array(o_final)

    def image_num_per_batch_train(tot_batch_size, attack_params):
        return max(1, tot_batch_size // attack_params.n_draws_attack)

    def image_num_per_batch_eval(tot_batch_size, attack_params):
        return max(1, tot_batch_size // attack_params.n_draws_eval)

