from typing import List

import torch
import torch.nn.functional as F
from .dice import to_tensor
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torch.nn.functional import softmax
import torch.nn as nn

from .functional import soft_jaccard_score

# __all__ = ["JaccardLoss", "BINARY_MODE", "MULTICLASS_MODE", "MULTILABEL_MODE"]
__all__ = ["JaccardLoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


# class JaccardLoss(_Loss):
#     """
#     Implementation of Jaccard loss for image segmentation task.
#     It supports binary, multi-class and multi-label cases.
#     """

#     def __init__(self, mode=MULTICLASS_MODE, classes: List[int] = None, log_loss=False, from_logits=True, smooth=0, eps=1e-7):
#         """

#         :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
#         :param classes: Optional list of classes that contribute in loss computation;
#         By default, all channels are included.
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param eps: Small epsilon for numerical stability
#         """
#         assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
#         super(JaccardLoss, self).__init__()
#         self.mode = mode
#         if classes is not None:
#             assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
#             classes = to_tensor(classes, dtype=torch.long)

#         self.classes = classes
#         self.from_logits = from_logits
#         self.smooth = smooth
#         self.eps = eps
#         self.log_loss = log_loss

#     def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
#         """

#         :param y_pred: NxCxHxW
#         :param y_true: NxHxW
#         :return: scalar
#         """
#         assert y_true.size(0) == y_pred.size(0)

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if self.mode == MULTICLASS_MODE:
#                 y_pred = y_pred.log_softmax(dim=1).exp()
#             else:
#                 y_pred = F.logsigmoid(y_pred).exp()

#         bs = y_true.size(0)
#         num_classes = y_pred.size(1)
#         dims = (0, 2)

#         if self.mode == BINARY_MODE:
#             y_true = y_true.view(bs, 1, -1)
#             y_pred = y_pred.view(bs, 1, -1)

#         if self.mode == MULTICLASS_MODE:
#             y_true = y_true.view(bs, -1)
#             y_pred = y_pred.view(bs, num_classes, -1)

#             y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
#             y_true = y_true.permute(0, 2, 1)  # H, C, H*W

#         if self.mode == MULTILABEL_MODE:
#             y_true = y_true.view(bs, num_classes, -1)
#             y_pred = y_pred.view(bs, num_classes, -1)

#         scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)

#         if self.log_loss:
#             loss = -torch.log(scores.clamp_min(self.eps))
#         else:
#             loss = 1.0 - scores

#         # IoU loss is defined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         mask = y_true.sum(dims) > 0
#         loss *= mask.float()

#         if self.classes is not None:
#             loss = loss[self.classes]

#         return loss.mean()


def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()
    
    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)
    
    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]
    
    return input_flatten, target_flatten


class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
      
    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            
            intersection = (input_c * target_c).sum()
            total = (input_c + target_c).sum()
            union = total - intersection
            IoU = (intersection + self.smooth)/(union + self.smooth)
            
            losses.append(1-IoU)
        
        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


# class JaccardLoss(nn.Module):
#     def __init__(self, ignore_index=255, smooth=1.0, class_weights=None):
#         super(JaccardLoss, self).__init__()
#         self.ignore_index = ignore_index
#         self.smooth = smooth
#         self.class_weights = class_weights  # 类别权重
    
#     def forward(self, input, target):
#         input, target = flatten(input, target, self.ignore_index)
#         input = F.softmax(input, dim=1)  # softmax计算概率
#         num_classes = input.size(1)
        
#         losses = []
#         for c in range(num_classes):
#             target_c = (target == c).float()
#             input_c = input[:, c]
            
#             # IoU计算
#             intersection = (input_c * target_c).sum()
#             total = (input_c + target_c).sum()
#             union = total - intersection
#             IoU = (intersection + self.smooth) / (union + self.smooth)
            
#             # 权重计算
#             weight = 1.0  # 默认权重为1
#             if self.class_weights is not None:
#                 weight = self.class_weights[c]
            
#             losses.append(weight * (1 - IoU))
        
#         losses = torch.stack(losses)
#         loss = losses.mean()
#         return loss
