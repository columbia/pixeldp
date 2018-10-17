figsize          = (7, 3.5)
legend_font_size = 14
ticks_font_size  = 12
labels_font_size = 15
line_thickness   = 3
markersize       = 7

def label(model, model_params):
    if type(model) == str and model == 'robustop':
        return "RobustOpt"
    name = model.__name__.split('.')[-1]
    if name == 'madry':
        return r'Madry'
        #  return r'Madry, L=0.03 ($\infty$-norm)'
    if model_params.attack_norm_bound <= 0:
        return 'Baseline'

    if model_params.attack_norm == 'l_inf':
        norm_p = '\infty'
    elif model_params.attack_norm == 'l1':
        norm_p = 1
    elif model_params.attack_norm == 'l2':
        norm_p = 2

    L = round(model_params.attack_norm_bound, 2)
    #  if L == 0.08: L = 0.1
    return 'PixelDP, L={} (${}$-norm)'.format(
        L,
        norm_p
    )

def color(model, model_params):
    if type(model) == str and model == 'robustop':
        return 'C1'
    name = model.__name__.split('.')[-1]
    if name == 'madry':
        return 'C1'
    if model_params.attack_norm_bound <= 0:
        return 'C3'

    if model_params.sensitivity_norm == 'l2':
        if model_params.noise_after_n_layers == 0:
            # image noise
            return 'C5'
        # Default PixelDP
        return'C0'
    elif model_params.sensitivity_norm == 'l1':
        if model_params.noise_after_n_layers == 0:
            # image noise
            return 'C6'
        # Default PixelDP
        return'C2'

    return "C7"

def linestyle(model, model_params):
    if type(model) == str and model == 'robustop':
        return '--'
    name = model.__name__.split('.')[-1]
    if name == 'madry':
        return '--'
    if model_params.attack_norm_bound <= 0:
        return '--'

    if model_params.noise_after_n_layers == 0:
        # image noise
        return '-.'

    L = round(model_params.attack_norm_bound, 2)
    if L == 0.08:
        return ':'
    # Default PixelDP
    return'-'

def markerstyle(model, model_params):
    if type(model) == str and model == 'robustop':
        return 'P'
    name = model.__name__.split('.')[-1]
    if name == 'madry':
        return 'P'
    #  if name == 'robust_opt':
        #  return 'v'
    if model_params.attack_norm_bound <= 0:
        return 'X'

    attack_bound = round(model_params.attack_norm_bound, 2)
    if attack_bound == 0.1:
        # Default PixelDP
        return 'o'
    elif attack_bound == 0.08:
        #  return 'o'
        return '>'
    elif attack_bound == 0.03:
        return 'D'
    elif attack_bound == 0.3:
        return '^'
    elif attack_bound == 1.0:
        return 's'

    #  remains: "h" "<" ">"

def robust_accuracy_survival_ps(curves_x, pred_truth, robustness):
    true_and_robust_n = sum(pred_truth)
    d      = sorted(zip(robustness, pred_truth))
    tot    = len(d)
    robust = tot
    robust_acc_survival_p = []
    for x in curves_x:
        obs_index = tot - robust
        while obs_index < tot and d[obs_index][0] < x:
            if d[obs_index][1]:
                true_and_robust_n -= 1
            robust -= 1
            obs_index += 1
        robust_acc_survival_p.append(true_and_robust_n / tot)

    return robust_acc_survival_p

def robust_prec_rec(curves_x, pred_truth, robustness):
    true_and_robust_n = sum(pred_truth)
    d      = sorted(zip(robustness, pred_truth))
    tot    = len(d)
    robust = tot
    robust_prec = []
    robust_prec_n = []
    for x in curves_x:
        obs_index = tot - robust
        while obs_index < tot and d[obs_index][0] < x:
            if d[obs_index][1]:
                true_and_robust_n -= 1
            robust -= 1
            obs_index += 1
        if robust > 0:
            robust_prec.append(true_and_robust_n / robust)
        robust_prec_n.append(robust / tot)

    return {
        'robust_prec':   robust_prec,
        'robust_prec_n': robust_prec_n,
    }

def accuracy(pred_truth):
    if len(pred_truth) == 0:
        return 0
    return sum(pred_truth) / len(pred_truth)

