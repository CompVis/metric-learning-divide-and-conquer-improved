import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from metriclearning.sampler import pdist
from metriclearning.sampler import topk_mask
from metriclearning.sampler import masked_maximum
from metriclearning.sampler import masked_minimum



class NPairsLoss(torch.nn.Module):
    """
    Computes the NPairs loss using Softmax.

    The loss takes each row of the pair-wise similarity matrix, `y_pred`,
    as logits and the remapped multi-class labels, `y_true`, as labels.

    Uses learnable scaling factor alpha (inverse of the temperature)

    See: https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf

    """
    def __init__(self, alpha=1.0, **kwargs):
        torch.nn.Module.__init__(self)
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float32))
        self.n_logged = 0

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
              be l2 normalized.
            labels: 1-D int32 `Tensor` with shape [batch_size] of
              multiclass integer labels.

        Returns:
            loss: float32 scalar.
        """
        sims = torch.mm(embeddings, embeddings.t())
        n = len(sims)
        pos = torch.eq(
            *[labels.unsqueeze(dim).expand_as(sims) for dim in [0, 1]]
        ).type_as(sims)

        # exclude self from the loss calculation
        mask = ~torch.eq(
            *[torch.arange(n).unsqueeze(dim).expand_as(sims) for dim in [0, 1]]
        )
        pos = pos[mask].reshape(n, n - 1)
        sims = sims[mask].reshape(n, n - 1)

        #drop rows without positives
        mask_rows_with_pos = pos.sum(dim=1).nonzero().squeeze()
        pos = pos[mask_rows_with_pos]
        sims = sims[mask_rows_with_pos] * self.alpha

        soft_labels =  pos / pos.sum(dim=1, keepdim=True)

        # Maybe temperature is required for the loss to work
        logits = F.log_softmax(sims, dim=1)
        loss = torch.mean(torch.sum(-logits * soft_labels, dim=1))
        #print(f'**npairs mean loss={loss:.4f}')
        #import ipdb; ipdb.set_trace()
        if self.n_logged == 0:
            logging.debug(f'npairs.alpha = {self.alpha.data.item()}')
        self.n_logged = (self.n_logged + 1) % 10
        return loss

