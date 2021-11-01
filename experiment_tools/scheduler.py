# https://github.com/ae-foster/pytorch-simclr/blob/simclr-master/scheduler.py
import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithConstantLRandLinearRampLR(_LRScheduler):
    def __init__(self, optimizer, T_min, T_max, eta_min=0, last_epoch=-1, ramp_len=10):
        self.T_max = T_max
        self.T_min = T_min
        self.eta_min = eta_min
        self.ramp_len = ramp_len
        super(CosineAnnealingWithConstantLRandLinearRampLR, self).__init__(
            optimizer, last_epoch
        )

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        cosine_lr = [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.T_min) / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
        linear_lr = [
            base_lr * (1 + (self.last_epoch - self.T_min)) / self.ramp_len
            for base_lr in self.base_lrs
        ]
        if self.last_epoch - self.T_min < 0:
            return self.base_lrs
        lr = [min(x, y) for x, y in zip(cosine_lr, linear_lr)]
        # print(lr)
        return lr
