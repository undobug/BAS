# --------------------------------------------------------
#!/usr/bin/env python
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


import math
from torch import optim as optim
import torch.optim.lr_scheduler as lr_scheduler

try:
    from apex.optimizers import FusedAdam, FusedLAMB
except:
    FusedAdam = None
    FusedLAMB = None
    print("To use FusedLAMB or FusedAdam, please install apex.")


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_adam':
        optimizer = FusedAdam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_lamb':
        optimizer = FusedLAMB(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def build_scheduler(config, optimizer):

    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        scale = 0.1
        lf = lambda x: ((1 + math.cos(x * math.pi / config.TRAIN.EPOCHS)) / 2) * (1 - scale) + scale  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.65)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'cosine_restart':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=3, eta_min=0, last_epoch=-1, verbose=False)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

    return scheduler


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin