from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class CosineWithWarmup(_LRScheduler):

    def __init__(
            self,
            optimizer,
            T_max,
            warmup_epochs,
            eta_min=0
    ):
        self.cosine_scheduler = CosineAnnealingLR(optimizer,
                                                  T_max - warmup_epochs,
                                                  eta_min)
        self.warmup_epochs = warmup_epochs
        self.finished = False
        super().__init__(optimizer=optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            return [
                base_lr / self.warmup_epochs for base_lr in self.base_lrs
            ]
        elif 0 < self.last_epoch < self.warmup_epochs:
            return [
                self.last_epoch * base_lr / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch == self.warmup_epochs:
            self.finished = True
            return self.base_lrs
        else:
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.cosine_scheduler.step(None)
            else:
                self.cosine_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super(CosineWithWarmup, self).step(epoch)


