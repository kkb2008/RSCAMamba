import torch
from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from .functional import label_smoothed_nll_loss

__all__ = ["SoftCrossEntropyLoss"]


# 这个损失函数没有被修改，是直接进行注释的,使用了下面的另外一个损失函数，与这个相比，只增加了一个class_weights参数。===2025年1月14日注释，当想在使用的时候直接取消注释就行了
class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )


# # ===2025年1月14日添加的===========
# class SoftCrossEntropyLoss(nn.Module):
#     """
#     Drop-in replacement for nn.CrossEntropyLoss with a few additions:
#     - Support for label smoothing
#     - Support for class-wise weighting
#     """

#     __constants__ = ["reduction", "ignore_index", "smooth_factor", "class_weights"]

#     def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0,
#                  ignore_index: Optional[int] = -100, dim=1, class_weights: Optional[torch.Tensor] = None):
#         """
#         :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
#         :param smooth_factor: The smoothing factor for label smoothing.
#         :param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
#         :param dim: The dimension over which softmax is applied.
#         :param class_weights: Optional tensor of weights for each class. Shape should be (C,), where C is the number of classes.
#         """
#         super().__init__()
#         self.smooth_factor = smooth_factor
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.dim = dim
#         self.class_weights = class_weights

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         """
#         :param input: The input logits (before softmax) of shape (N, C, *), where C is the number of classes.
#         :param target: The target labels of shape (N, *), with integer class indices.
#         :return: The computed loss (scalar or tensor depending on reduction).
#         """
#         log_prob = F.log_softmax(input, dim=self.dim)

#         # =======================这个区间中的代码都是新添加的=2025.1.16====================
#         b, c, h, w = log_prob.shape

#         # Expand weights_list to [b, c, 1, 1] on the same device
#         weights_expanded = self.class_weights.view(1, c, 1, 1)
        
#         # Broadcast to the target shape [b, c, h, w] on the same device
#         weights_tensor = weights_expanded.expand(b, c, h, w)
#         log_prob = log_prob * weights_tensor
#         # =======================这个区间中的代码都是新添加的=2025.1.16====================
        
#         # Call the label_smoothed_nll_loss function to get the base loss
#         loss = label_smoothed_nll_loss(
#             log_prob,
#             target,
#             epsilon=self.smooth_factor,
#             ignore_index=self.ignore_index,
#             reduction=self.reduction,
#             dim=self.dim,
#         )
#         return loss
