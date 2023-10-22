import torch

from .class_balanced_loss import CBLoss
from .focalloss import FocalLoss

class CombineLoss(torch.nn.Module):

    def __init__(self, samples_per_cls) -> None:
        super().__init__()
        self.cbl = CBLoss(samples_per_cls=samples_per_cls)
        self.fl = FocalLoss()

    def forward(self, outputs, targets):
        return self.cbl(outputs, targets) + self.fl(outputs, targets)

