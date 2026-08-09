"""Explainer interface and the Explanation container.

Hard constraint of the framework: every Explanation carries a `faithfulness`
slot — an explanation without verification scores is considered incomplete
("没有验证的解释不输出").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Explanation:
    """Attribution result on a spatial graph."""

    node_scores: np.ndarray            # (n,) or (n, F) gene-level attribution
    edge_scores: np.ndarray | None = None  # (E,) neighborhood/interaction edges
    faithfulness: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return len(self.faithfulness) > 0


class Explainer(ABC):
    @abstractmethod
    def explain(self, adapter: Any, data: Any, target: Any, **kwargs) -> Explanation:
        """Produce an Explanation for `target` on `data` via `adapter`."""
