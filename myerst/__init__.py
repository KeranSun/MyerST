"""MyerST: topology-constrained game-theoretic attribution for spatial transcriptomics."""

__version__ = "0.1.0"

from myerst.data.graph import build_knn_graph, adjacency_list, connected_components
from myerst.data.curvature import forman_ricci
from myerst.attribution.myerson import exact_myerson, mc_myerson
from myerst.adapters.base import ExplanationTarget
from myerst.adapters.torch_adapter import TorchModelAdapter
from myerst.explainers.ig import IGExplainer
from myerst.explainers.occlusion import SpatialOcclusion
from myerst.explainers.myerson_explainer import MyersonExplainer
from myerst.explainers.baselines import GraphLIME, GNNExplainerST
from myerst.explainers.local_ccc_fast import local_ccc_synergy_fast
from myerst.benchmark.simulator import LayeredTissueSimulator
from myerst.benchmark.ccc_simulator import CCCSimulator
from myerst.benchmark.faithfulness import FaithfulnessEvaluator
from myerst.benchmark.roar import ROAREvaluator

__all__ = [
    "build_knn_graph",
    "adjacency_list",
    "connected_components",
    "forman_ricci",
    "exact_myerson",
    "mc_myerson",
    "ExplanationTarget",
    "TorchModelAdapter",
    "IGExplainer",
    "SpatialOcclusion",
    "MyersonExplainer",
    "GraphLIME",
    "GNNExplainerST",
    "local_ccc_synergy_fast",
    "LayeredTissueSimulator",
    "CCCSimulator",
    "FaithfulnessEvaluator",
    "ROAREvaluator",
]
