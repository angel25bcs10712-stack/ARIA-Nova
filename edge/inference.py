"""
ARIA Nova Inference Backend Abstraction
Defines the base interface for hardware inference backends (CPU and QNN).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from edge.config import config
from edge.device import get_device_info


class InferenceBackend(ABC):
    """Abstract base class for all on-device inference backends."""

    def __init__(self, name: str, accelerator: str):
        self.name = name
        self.accelerator = accelerator
        self.is_loaded = False
        self.fallback_occurred = False
        self.fallback_reason: Optional[str] = None
        self.active_provider: Optional[str] = None

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load model weights or ONNX session into memory."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2
    ) -> str:
        """Generate text completion from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend hardware runtime is available."""
        pass

    def get_telemetry(self) -> Dict[str, Any]:
        """Return backend status and hardware execution info."""
        return {
            "backend": self.name,
            "accelerator": self.accelerator,
            "is_loaded": self.is_loaded,
            "fallback_occurred": self.fallback_occurred,
            "fallback_reason": self.fallback_reason,
            "active_provider": self.active_provider,
        }


def get_backend(backend_type: Optional[str] = None) -> InferenceBackend:
    """
    Factory function to retrieve the appropriate inference backend.
    
    Args:
        backend_type: "auto", "cpu", or "qnn". Defaults to config.backend_mode.
    """
    mode = (backend_type or config.backend_mode).lower()

    from edge.cpu_backend import CPUBackend
    from edge.qnn_backend import QNNBackend

    if mode == "qnn":
        return QNNBackend()
    elif mode == "cpu":
        return CPUBackend()
    elif mode == "auto":
        dev = get_device_info()
        if dev["qnn_available"] and dev["npu_available"]:
            return QNNBackend()
        return CPUBackend()
    else:
        # Default fallback
        return CPUBackend()
