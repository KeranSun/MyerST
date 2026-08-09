"""Faithful PyTorch port of the official STAGATE (Dong & Zhang, Nat Commun 2022).

The official implementation is TensorFlow-1.x (vendor/stagate/model.py); TF1
does not run on modern Python, so this module ports the exact architecture:

- hidden_dims [F, 512, 30], alpha=0 (single attention graph, no pruning)
- tied decoder weights (decoder uses W^T)
- per-layer single-head attention computed AFTER projection:
      f1 = A * (M v0); f2 = A * (M v1)^T; C = sparse_softmax(sigmoid(f1+f2))
  (dense equivalent: sigmoid, mask non-edges, softmax over source j per node i)
- no attention and no ELU on the last encoder layer; decoder mirrors encoder,
  ELU on all but the output layer
- loss = ||X - X_hat||_F (Frobenius) + weight_decay * L2(W_last)
- training: Adam lr=1e-4, gradient clipping 5, 500 epochs

Dense attention is used (n^2 matrix) — fine for Visium-scale slices.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STAGATEOfficialPT(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 512, latent: int = 30,
                 weight_decay: float = 1e-4):
        super().__init__()
        self.weight_decay = weight_decay
        self.W0 = nn.Parameter(torch.empty(n_feat, hidden))
        self.W1 = nn.Parameter(torch.empty(hidden, latent))
        self.v0 = nn.Parameter(torch.empty(hidden, 1))
        self.v1 = nn.Parameter(torch.empty(hidden, 1))
        for p in [self.W0, self.W1, self.v0, self.v1]:
            nn.init.xavier_uniform_(p)

    def attention(self, adj_bin: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Sparse softmax(sigmoid(f1+f2)) over edges; adj_bin has self-loops.
        Supports (n, F) and batched (B, n, F) inputs."""
        A = adj_bin
        while A.dim() < M.dim():
            A = A.unsqueeze(0)
        f1 = A * (M @ self.v0)                       # (..., n_i, n_j) row broadcast
        f2 = A * (M @ self.v1).transpose(-1, -2)
        s = torch.sigmoid(f1 + f2)
        s = s.masked_fill(A == 0, torch.finfo(s.dtype).min)
        return torch.softmax(s, dim=-1)              # over source j per node i

    def forward(self, x: torch.Tensor, adj_bin: torch.Tensor):
        # encoder
        H = x @ self.W0
        C = self.attention(adj_bin, H)
        H = F.elu(C @ H)
        Z = H @ self.W1                              # latent embedding
        # decoder (tied weights, attention reused)
        Hd = Z @ self.W1.T
        Hd = F.elu(C @ Hd)
        Xr = Hd @ self.W0.T
        return Xr, Z, C

    def embed(self, x: torch.Tensor, adj_bin: torch.Tensor) -> torch.Tensor:
        return self.forward(x, adj_bin)[1]

    def loss(self, x: torch.Tensor, adj_bin: torch.Tensor) -> torch.Tensor:
        Xr, _, _ = self.forward(x, adj_bin)
        fro = torch.sqrt(torch.sum((x - Xr) ** 2))
        return fro + self.weight_decay * torch.sum(self.W1 ** 2)


def train_stagate_official(model: STAGATEOfficialPT, x: torch.Tensor,
                           adj_bin: torch.Tensor, epochs: int = 500,
                           lr: float = 1e-4, grad_clip: float = 5.0,
                           seed: int = 2020, verbose: bool = False) -> STAGATEOfficialPT:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = model.loss(x, adj_bin)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if verbose and (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:4d}  loss {loss.item():.2f}", flush=True)
    model.eval()
    return model
