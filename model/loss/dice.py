import torch.nn as nn
from kornia.losses import dice_loss

class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, input, target):
        target = target.squeeze(1)
        loss = dice_loss(input, target)

        return loss

             
"""
class DICELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DICELoss, self).__init__()
        self.eps = eps

    def to_one_hot(self, target):
        N, C, H, W = target.size()
        assert C == 1
        target = torch.zeros(N, 2, H, W).to(target.device).scatter_(1, target, 1)
        return target

    def forward(self, input, target):
        N, C, _, _ = input.size()
        input = F.softmax(input, dim=1)

        #target = self.to_one_hot(target)
        target = torch.eye(2)[target.squeeze(1)]
        target = target.permute(0, 3, 1, 2).type_as(input)

        dims = tuple(range(1, target.ndimension()))
        inter = torch.sum(input * target, dims)
        cardinality = torch.sum(input + target, dims)
        loss = ((2. * inter) / (cardinality + self.eps)).mean()

        return 1 - loss
"""


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, smooth=1e-5):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        """
        input: Tensor of shape (B, C, H, W) - probabilities or logits (after softmax)
        target: Tensor of shape (B, H, W) - class indices
        """
        if input.size(1) == 1:
                                  
            input = input.squeeze(1)
            target = target.float()
            inter = (input * target).sum()
            dice = (2. * inter + self.smooth) / (input.sum() + target.sum() + self.smooth)
            return 1 - dice

                                               
        input = F.softmax(input, dim=1)

        total_loss = 0.0
        num_classes = input.size(1)

        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            pred_flat = input[:, c, :, :].contiguous().view(-1)
            target_flat = (target == c).float().view(-1)

            intersection = (pred_flat * target_flat).sum()
            dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

                                         
                                                          

                                          
            class_loss = 1 - dice_score
            if self.weight is not None:
                class_loss = class_loss * self.weight[c]                           

            total_loss += class_loss
        return total_loss / num_classes
