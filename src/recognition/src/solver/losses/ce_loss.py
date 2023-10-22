import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss


class CELoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, stu_pred, label):
        loss = F.cross_entropy(stu_pred, label)
        return loss