"""STAGATE-lite: a pure-PyTorch graph attention autoencoder.

Faithful in spirit to STAGATE (Dong & Zhang, Nat Commun 2022) — GAT encoder/
decoder reconstructing expression from the spatial graph — but simplified:
dense attention, fewer heads, no PyG dependency. Serves as the second host
architecture to demonstrate model-agnostic explanation (and later provides
edge attention weights as the falsifiable baseline in R3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Dense multi-head graph attention layer (adjacency-masked)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, alpha: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj_bin: torch.Tensor,
                return_attention: bool = False):
        # supports (n, F) or (..., n, F) batched inputs
        h = self.W(x).view(*x.shape[:-1], self.heads, self.out_dim)  # (..., n, H, D)
        e_src = (h * self.a_src).sum(-1)                             # (..., n, H)
        e_dst = (h * self.a_dst).sum(-1)
        e = self.leaky(e_src.unsqueeze(-3) + e_dst.unsqueeze(-2))    # (..., n_i, n_j, H)
        mask = adj_bin.unsqueeze(-1) > 0                             # (n, n, 1), broadcasts
        e = e.masked_fill(~mask, torch.finfo(e.dtype).min)
        attn = F.softmax(e, dim=-2)                                  # over source j
        out = torch.einsum("...ijh,...jhd->...ihd", attn, h)         # (..., n, H, D)
        out = out.reshape(*x.shape[:-1], self.heads * self.out_dim)
        if return_attention:
            return out, attn.mean(-1)                                # mean over heads
        return out


class STAGATELite(nn.Module):
    """GAT autoencoder: encoder GAT(in->hidden->latent), decoder mirrors it."""

    def __init__(self, n_feat: int, hidden: int = 64, latent: int = 32, heads: int = 2):
        super().__init__()
        self.enc1 = GATLayer(n_feat, hidden, heads=heads)
        self.enc2 = GATLayer(hidden * heads, latent, heads=1)
        self.dec1 = GATLayer(latent, hidden, heads=heads)
        self.dec2 = GATLayer(hidden * heads, n_feat, heads=1)

    def embed(self, x: torch.Tensor, adj_bin: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.enc1(x, adj_bin))
        return self.enc2(h, adj_bin)

    def forward(self, x: torch.Tensor, adj_bin: torch.Tensor) -> torch.Tensor:
        z = self.embed(x, adj_bin)
        h = F.elu(self.dec1(z, adj_bin))
        return self.dec2(h, adj_bin)


def train_stagate(model: STAGATELite, x: torch.Tensor, adj_bin: torch.Tensor,
                  epochs: int = 300, lr: float = 0.005, weight_decay: float = 1e-4,
                  seed: int = 0, verbose: bool = False) -> STAGATELite:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(x, adj_bin), x)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:3d}  recon_mse {loss.item():.5f}")
    model.eval()
    return model


class HeadOnEmbedding(nn.Module):
    """Logistic head on frozen STAGATE embeddings — makes an unsupervised host
    explainable with the same (x, adj) -> logits adapter contract."""

    def __init__(self, backbone: STAGATELite, n_classes: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(backbone.enc2.out_dim, n_classes)

    def forward(self, x: torch.Tensor, adj_bin: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.embed(x, adj_bin))


def adj_binary(edges: np.ndarray, n_nodes: int) -> torch.Tensor:
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if len(edges):
        A[edges[:, 0], edges[:, 1]] = 1.0
        A[edges[:, 1], edges[:, 0]] = 1.0
    A += np.eye(n_nodes, dtype=np.float32)
    return torch.from_numpy(A)
