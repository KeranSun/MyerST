"""Minimal pure-PyTorch GCN host model (no PyG dependency).

Deliberately dependency-free: PyG's compiled extensions are fragile on
Windows + new Python versions, and a 2-layer GCN is ~30 lines. Serves as the
reference host model for the attribution experiments (STAGATE/GraphST
adapters follow the same contract).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(edges: np.ndarray, n_nodes: int) -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2 (dense).

    Dense is fine for Visium-scale graphs (~5k spots); sparse variant can be
    added when moving to Xenium-scale data.
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if len(edges):
        A[edges[:, 0], edges[:, 1]] = 1.0
        A[edges[:, 1], edges[:, 0]] = 1.0
    A += np.eye(n_nodes, dtype=np.float32)
    deg = A.sum(1)
    dinv = np.zeros_like(deg)
    np.power(deg, -0.5, out=dinv, where=deg > 0)
    A_norm = dinv[:, None] * A * dinv[None, :]
    return torch.from_numpy(A_norm.astype(np.float32))


class GCN(nn.Module):
    def __init__(self, n_feat: int, n_hidden: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.w1 = nn.Linear(n_feat, n_hidden)
        self.w2 = nn.Linear(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(adj_norm @ self.w1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return adj_norm @ self.w2(h)


def train_gcn(
    model: GCN,
    x: torch.Tensor,
    adj_norm: torch.Tensor,
    labels: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 0,
    verbose: bool = False,
    return_mask: bool = False,
):
    """Full-batch training with a stratified 80/20 split; returns trained model
    (plus train mask when return_mask=True — needed for held-out ROAR eval)."""
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
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(x, adj_norm)
        loss = F.cross_entropy(logits[tr], y[tr])
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x, adj_norm).argmax(1)[~tr] == y[~tr]).float().mean().item()
            model.train()
            print(f"  epoch {ep+1:3d}  loss {loss.item():.4f}  val_acc {acc:.4f}")
    model.eval()
    if return_mask:
        return model, train_mask
    return model
