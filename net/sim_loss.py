import torch
import torch.nn.functional as F
import torch.nn as nn


class CosineSimLoss(nn.Module):
    """NLL Loss.
    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self,
                 with_norm=True,
                 negative=False,
                 pairwise=False,
                 loss_weight=1.0):

        super().__init__()
        self.with_norm = with_norm
        self.negative = negative
        self.pairwise = pairwise
        self.loss_weight = loss_weight

    def forward(self, *args):
        """Defines the computation performed at every call.
        Args:
            *args: The positional arguments for the corresponding
                loss.
        Returns:
            torch.Tensor: The calculated loss.
        """
        return self._forward(*args) * self.loss_weight

    def _forward(self, cls_score, label, mask=None, **kwargs):
        if self.with_norm:
            cls_score = F.normalize(cls_score, p=2, dim=1)
            label = F.normalize(label, p=2, dim=1)
        if mask is not None:
            assert self.pairwise
        if self.pairwise:
            cls_score = cls_score.flatten(2)
            label = label.flatten(2)
            prod = torch.einsum('bci,bcj->bij', cls_score, label)
            if mask is not None:
                assert prod.shape == mask.shape
                prod *= mask.float()
            prod = prod.flatten(1)
        else:
            prod = torch.sum(
                cls_score * label, dim=1).view(cls_score.size(0), -1)
        if self.negative:
            loss = -prod.mean(dim=-1)
        else:
            loss = 2 - 2 * prod.mean(dim=-1)
        return loss
