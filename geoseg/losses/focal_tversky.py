import torch
import torch.nn as nn
from torch.nn.functional import softmax


def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()
    
    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)
    
    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]
    
    return input_flatten, target_flatten


class FocalTverskyLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0, alpha=0.5, beta=0.5, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            
            t_p = (input_c * target_c).sum()
            f_p = ((1-target_c) * input_c).sum()
            f_n = (target_c * (1-input_c)).sum()
            tversky = (t_p + self.smooth) / (t_p + self.alpha*f_p + self.beta*f_n + self.smooth)
            focal_tversky = (1 - tversky)**self.gamma
            losses.append(focal_tversky)
        
        losses = torch.stack(losses)
        loss = losses.mean()
        return loss