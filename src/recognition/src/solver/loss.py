import logging
import torch.nn as nn


from .losses import *
from config import cfg

__all__ = ['get_loss','mixup_criterion']


def CrossEntropy():
    return CELoss()


def focalloss(Gamma = cfg.OPTIM.FOCAL_GAMMA, Reduction='mean'):
    return FocalLoss(gamma=Gamma,reduction=Reduction)


def LabelSmoothCE(labelsmooth = cfg.OPTIM.LABEL_SMOOTH,class_number=cfg.DATASET.CLASS_NUMS):
    return LabelSmoothingLoss(label_smoothing=labelsmooth,class_nums=class_number)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def combine_loss(samples_per_cls=[]):
    return CombineLoss(samples_per_cls)

losses = {
    'ce':CrossEntropy,
    'focal':focalloss,
    'labelsmoothce':LabelSmoothCE,
    'combine': combine_loss, 

}


def get_loss(loss_name=cfg.OPTIM.LOSS,**kwargs):
    loss_name = loss_name.lower()
    if loss_name in losses:
        return losses[loss_name](**kwargs).cuda()
    else:
        print("Loss: {} is not implemented...,\
            using default loss: CrossEntropyLoss".format(loss_name))
        return nn.CrossEntropyLoss()