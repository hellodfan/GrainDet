import os, cv2
import numpy as np

import torch

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


def filter_box(splitsize, per_img_boxes, pos, rows, cols):
    gap = 10
    x1 = per_img_boxes[:,0]
    y1 = per_img_boxes[:,1]
    x2 = per_img_boxes[:,2]
    y2 = per_img_boxes[:,3]

    mask = [False]*len(per_img_boxes)
    px,py = pos[0], pos[1]
    mask1 = x1 < gap
    mask2 = y1 < gap
    mask3 = x2 > splitsize - gap
    mask4 = y2 > splitsize - gap
    
    if px<gap and gap<py<rows-splitsize-gap:
        mask =  mask2 + mask3 + mask4
    
    if px>cols-splitsize-gap and gap<py<rows-splitsize-gap:
        mask =  mask1 + mask2 + mask4

    if py<gap and gap<px<cols-splitsize-gap:
        mask =  mask1 + mask3 + mask4

    if py>rows-splitsize-gap and gap<px<cols-splitsize-gap:
        mask =  mask1 + mask2 + mask3

    if px<gap and py<gap:
        mask = mask3 + mask4

    if px<gap and py>rows-splitsize-gap:
        mask = mask2 + mask3

    if px>cols-splitsize-gap and py>rows-splitsize-gap:
        mask = mask1 + mask2

    if px>cols-splitsize-gap and py<gap:
        mask = mask1 + mask4

    if gap<px<cols-splitsize-gap and gap<py<rows-splitsize-gap:
        mask = mask1 + mask2 + mask3 + mask4

    mask =  ~mask

    return mask


def compute_IOU(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S1 + S2 - S_cross)


def get_f1(all_test, all_gt):

    # count_gt为标注的所有数据框
    count_gt = {}
    # count_test为检测的所有数据框
    count_test = {}
    # count_yes_test为检测正确的数据框
    count_yes_test = {}
    # count_no_test为检测错误的数据框
    count_no_test = {}


    # 下面主要思想：遍历test结果，再遍历对应gt的结果，如果两个框的iou大于一定的阙址并且类别相同，视为正确
    for ix, f_test_lines in enumerate(all_test):
        f_gt_lines = all_gt[ix]
        for f_test_line in f_test_lines:
            flag = 1
            for f_gt_line in f_gt_lines:
                gt_label, gt_xmin, gt_ymin, gt_xmax, gt_ymax = f_gt_line[0], f_gt_line[1], f_gt_line[2], f_gt_line[3], f_gt_line[4]
                test_label,test_xmin, test_ymin, test_xmax, test_ymax = f_test_line[0], f_test_line[1], f_test_line[2], f_test_line[3], f_test_line[4]
                test_score = f_test_line[5]
                IOU = compute_IOU([gt_xmin, gt_ymin, gt_xmax, gt_ymax],
                                [test_xmin, test_ymin, test_xmax, test_ymax])
                if gt_label == test_label and IOU >= 0.5 and test_score >= 0.3:
                    flag = 0
                    if f_test_line[0] not in count_yes_test:
                        count_yes_test[f_test_line[0]] = 0
                    count_yes_test[f_test_line[0]] += 1
    
            if flag == 1:
                if f_test_line[0] not in count_no_test:
                    count_no_test[f_test_line[0]] = 0
                count_no_test[f_test_line[0]] += 1
                
    # 有以下4个结果，就可以计算相关指标了
    print(count_gt)
    print(count_test)
    print(count_yes_test)
    print(count_no_test)



# NMS算法
# bboxes维度为[N,4]，scores维度为[N,], 均为tensor
def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor