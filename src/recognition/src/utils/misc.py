
import os
import torch
import shutil
import logging
from thop import profile
import numpy as np
import torchvision.transforms as transforms
# from datasets.reader import get_imglists
# from config import cfg

__all__ = ['accuracy','AverageMeter','show_flops_params']



# def get_transform_6c():
#     transform6 = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cfg.DATASET.MEANS, std=cfg.DATASET.STD),
#     ])
#     return transform6


# def get_dataloader(img_path, mode='train',spilt='train'):
#     test_imgs = get_imglists(img_path, mode=data_mode, spilt=data_mode)
#     test_dataset = WheatchannelDataset(test_imgs, mode=data_mode, transforms=transform)
#     test_loader = torch.utils.data.DataLoader(test_dataset,
#                                              batch_size=batch_size, shuffle=True,
#                                              num_workers=12, pin_memory=True, drop_last=False)
#     return test_loader







def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class Precision:
    """
    Computes precision of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of precision score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + self.epsilon)
        return precision


class Recall:
    """
    Computes recall of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of recall score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        actual_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = true_positives / (actual_positives + self.epsilon)
        return recall


class F1Score:
    """
    Computes F1-score between `y_true` and `y_pred`.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of F1-score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon
        self.precision = Precision()
        self.recall = Recall()

    def __call__(self, y_pred, y_true):
        precision = self.precision(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        return 2 * ((precision * recall) / (precision + recall + self.epsilon))


def confusion_matrix(preds_list, lables_list, CLASS_NUMS):
    """
    brief:统计模型识别结果（对于已经有明确label的样本）
     :
     :return:返回统计结果
    """

    confusion =  np.zeros((CLASS_NUMS, CLASS_NUMS), dtype=np.int) # 统计结果

    for i in range(len(lables_list)):
        if lables_list[i] == preds_list[i]:
            confusion[lables_list[i]][lables_list[i]] += 1
        else:
            confusion[lables_list[i]][preds_list[i]] += 1
    return confusion



class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def show_flops_params(model, device, input_shape=[1, 3, 256, 384]):
    h,w =224,224
    c = 3
    input_shape = [1,c,w,h]
    #summary(model, tuple(input_shape[1:]), device=device)
    input = torch.randn(*input_shape).to(torch.device(device))
    flops, params = profile(model, inputs=(input,), verbose=False)

    logging.info('{} flops: {:.3f}G input shape is {}, params: {:.3f}M'.format(
        model.__class__.__name__, flops / 1000000000, input_shape[1:], params / 1000000))


# f1 = F1Score()
# x = torch.randn(32,7)
# y = torch.randint(0,6,(32,1)).reshape(-1)

# res = f1(x.argmax(-1).float(),y.float())
# q=1