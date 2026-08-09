"""GAT classifier host — needed for the attention-weight baseline explainer.

GCN has no attention, so the "attention as explanation" baseline (the one we
falsify in the paper) needs a GAT host. Single GATLayer + linear head, with
the attention matrix exposed for the baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from myerst.models.stagate_lite import GATLayer


class GATClassifier(nn.Module):
    def __init__(self, n_feat: int, n_hidden: int = 32, n_classes: int = 3,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.gat = GATLayer(n_feat, n_hidden, heads=heads)
        self.lin = nn.Linear(n_hidden * heads, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_bin: torch.Tensor,
                return_attention: bool = False):
        if return_attention:
            h, attn = self.gat(x, adj_bin, return_attention=True)
            h = F.dropout(F.elu(h), p=self.dropout, training=self.training)
            return self.lin(h), attn
        h = F.elu(self.gat(x, adj_bin))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)


def train_gat(model: GATClassifier, x: torch.Tensor, adj_bin: torch.Tensor,
              labels: np.ndarray, epochs: int = 200, lr: float = 0.01,
              weight_decay: float = 5e-4, seed: int = 0,
              return_mask: bool = False):
    torch.manual_seed(seed)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    n = len(y)
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(n, dtype=bool)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        train_mask[idx[: int(0.8 * len(idx))]] = True
    tr = torch.from_numpy(train_mask)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x, adj_bin)[tr], y[tr])
        loss.backward()
        opt.step()
    model.eval()
    if return_mask:
        return model, train_mask
    return model
