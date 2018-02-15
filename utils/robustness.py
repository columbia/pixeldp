import numpy as np
import math
import scipy.stats

# UB and LB from:
#   https://gist.github.com/DavidWalz/8538435
#   http://www.statsmodels.org/dev/_modules/statsmodels/stats/proportion.html#proportion_confint
# using the Clopper–Pearson interval:
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
def one_sided_upper_bound(count, nobs, alpha, bonferroni_hyp_n=1):
    return scipy.stats.beta.ppf(1 - alpha/bonferroni_hyp_n, count+1, nobs-count)
def one_sided_lower_bound(count, nobs, alpha, bonferroni_hyp_n=1):
    return scipy.stats.beta.ppf(alpha/bonferroni_hyp_n, count, nobs-count+1)

def _laplace_robustness_size(counts, attack_size, dp_epsilon):
    hyp_n = len(counts)
    count_tot = sum(counts)
    count2, count1 = sorted(counts)[-2:]
    p_max = one_sided_lower_bound(count1, count_tot, 0.05, bonferroni_hyp_n=hyp_n)
    p_sec = one_sided_upper_bound(count2, count_tot, 0.05, bonferroni_hyp_n=hyp_n)

    if p_max <= p_sec:
        # we're not even robust to the measurement error...
        return 0.0

    return attack_size * math.log(p_max/p_sec) / (2 * dp_epsilon)

def _guaussian_mech_mult(delta):
    return math.sqrt(2 * math.log(1.25 / delta))

def _gaussian_robustness_size(counts, attack_size, dp_epsilon, dp_delta):
    hyp_n = len(counts)
    count_tot = sum(counts)
    count2, count1 = sorted(counts)[-2:]
    p_max = one_sided_lower_bound(count1, count_tot, 0.05, bonferroni_hyp_n=hyp_n)
    p_sec = one_sided_upper_bound(count2, count_tot, 0.05, bonferroni_hyp_n=hyp_n)

    if p_max - dp_delta <= p_sec + dp_delta:
        # we're not even robust to the measurement error...
        return 0.0

    max_r = 0.0
    max_r_eps  = None
    max_r_delt = None
    delta_range = list(np.arange(0.001, 0.3, 0.001))
    #  epsilon_range = list(np.arange(0.1, 1.00000001, 0.001))  # we want 1 included
    for delta in delta_range:
        eps_min, eps_max, eps = (0.0, 1.0, 0.5)
        while eps_min < eps and eps_max >= eps:
        #  for eps in epsilon_range:
            l = attack_size *  \
                (eps / dp_epsilon) *  \
                (_guaussian_mech_mult(dp_delta) / _guaussian_mech_mult(delta))
            if p_max >= math.e ** (2 * eps) * p_sec + (1 + math.e ** eps) * delta:
                if l > max_r:
                    max_r = l
                    max_r_eps = eps
                    max_r_delt = delta
                # best eps for this delta may be bigger
                eps_min = eps
                eps = (eps_min + eps_max) / 2.0
            else:
                # eps is too big for delta
                eps_max = eps
                eps = (eps_min + eps_max) / 2.0

            if eps_max - eps_min < 0.001:
                break

    print(max_r_eps)
    print(max_r_delt)
    print(max_r)
    print()
    return max_r

def robustness_size(counts, dp_attack_size, dp_epsilon, dp_delta, dp_mechanism):
    if   dp_mechanism == 'laplace':
        return _laplace_robustness_size(counts, dp_attack_size, dp_epsilon)
    elif dp_mechanism == 'gaussian':
        return _gaussian_robustness_size(counts, dp_attack_size, dp_epsilon, dp_delta)
    else:
        raise ValueError('Only supports the following DP mechanisms: laplace gaussian.')

