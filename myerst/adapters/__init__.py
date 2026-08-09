from myerst.adapters.base import HostModelAdapter, ExplanationTarget

try:
    from myerst.adapters.torch_adapter import TorchModelAdapter
    __all__ = ["HostModelAdapter", "ExplanationTarget", "TorchModelAdapter"]
except ImportError:  # torch not installed
    __all__ = ["HostModelAdapter", "ExplanationTarget"]
