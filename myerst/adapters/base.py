"""Host model adapter interface.

Any host model (GCN/GAT/STAGATE/GraphST/NicheCompass/user-defined) is wrapped
behind this minimal contract so explainers stay model-agnostic. Torch is only
required by concrete adapters, never by this base module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExplanationTarget:
    """What to explain.

    kind: "domain_boundary" | "communication" | "embedding_dim"
    payload: e.g. (domainA, domainB), (sender_ct, receiver_ct, lr_pair), or int
    """

    kind: str
    payload: Any


class HostModelAdapter(ABC):
    """Uniform attributable interface for arbitrary host models."""

    @abstractmethod
    def forward(self, data: Any) -> Any:
        """Model output (logits or embeddings) for the given graph data."""

    @abstractmethod
    def target_output(self, data: Any, target: ExplanationTarget) -> Any:
        """Scalarized output for attribution (must support autograd)."""

    def parameters_trainable(self) -> bool:
        """Whether the adapter exposes a trainable head (default: no)."""
        return False
