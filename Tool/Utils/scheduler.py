"""
LR scheduler builder with warmup support.

Key fixes from the original implementation:
  - Respects cfg['type'] ('cos' or 'step') instead of always using LambdaLR.
  - Warmup epochs and init_lr are read from config, not hardcoded.
  - CosineAnnealingLR T_max accounts for warmup epochs so the main scheduler
    starts at the end of warmup, producing a continuous LR curve.
  - Cosine supports eta_min (min_lr) from config.

CosineWarmupLR / LinearWarmupLR / BNMomentumScheduler are preserved with
unchanged interfaces.
"""
import math
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched


def build_lr_scheduler(cfg, optimizer, last_epoch):
    """
    Args:
        cfg:
            type:           'cos' | 'cosine' | 'step' | 'multistep'
            warmup:         bool
            warmup_epochs:  int (default 3)
            warmup_init_lr: float (default 1e-5)
            min_lr:         float (default 1e-7) -- cosine eta_min
            max_epoch:      int -- required for cosine (= trainer.max_epoch)
            decay_list:     list[int] -- for multistep
            decay_rate:     float -- for multistep
        optimizer: torch.optim.Optimizer
        last_epoch: int, epoch-1 for resume, otherwise -1

    Returns:
        (main_scheduler, warmup_scheduler) -- warmup_scheduler may be None.
    """
    sched_type = str(cfg.get('type', 'step')).lower()
    warmup_enabled = bool(cfg.get('warmup', False))
    warmup_epochs = int(cfg.get('warmup_epochs', 3))
    warmup_init_lr = float(cfg.get('warmup_init_lr', 1e-5))
    min_lr = float(cfg.get('min_lr', 1e-7))

    if sched_type in ('cos', 'cosine'):
        max_epoch = int(cfg.get('max_epoch', 70))
        effective_warmup = warmup_epochs if warmup_enabled else 0
        T_max = max(max_epoch - effective_warmup, 1)
        lr_scheduler = lr_sched.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=min_lr, last_epoch=last_epoch)

    elif sched_type in ('step', 'multistep'):
        decay_list = list(cfg.get('decay_list', []))
        decay_rate = float(cfg.get('decay_rate', 0.1))

        def lr_lbmd(cur_epoch):
            cur_decay = 1.0
            for decay_step in decay_list:
                if cur_epoch >= decay_step:
                    cur_decay *= decay_rate
            return cur_decay

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    else:
        raise ValueError(
            f"Unknown lr_scheduler type: {sched_type!r}. "
            "Supported: 'cos' / 'cosine' / 'step' / 'multistep'.")

    warmup_lr_scheduler = None
    if warmup_enabled and warmup_epochs > 0:
        warmup_lr_scheduler = CosineWarmupLR(
            optimizer, num_epoch=warmup_epochs, init_lr=warmup_init_lr)
    return lr_scheduler, warmup_lr_scheduler


def build_bnm_scheduler(cfg, model, last_epoch):
    if not cfg['enabled']:
        return None

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return max(cfg['momentum'] * cur_decay, cfg['clip'])

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"Class '{type(model).__name__}' is not a PyTorch nn Module")
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.init_lr + (base_lr - self.init_lr)
            * (1 - math.cos(math.pi * self.last_epoch / self.num_epoch)) / 2
            for base_lr in self.base_lrs]


class LinearWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.init_lr + (base_lr - self.init_lr) * self.last_epoch / self.num_epoch
            for base_lr in self.base_lrs]
