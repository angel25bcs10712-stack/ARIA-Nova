"""
ARIA Nova Model Manager
Manages local small language model lifecycle, tokenizer, and backend binding.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from edge.config import config
from edge.inference import get_backend, InferenceBackend
from edge.device import get_device_info


class ModelManager:
    """Central manager for on-device SLM inference."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.model_path
        self.backend: Optional[InferenceBackend] = None
        self.is_ready = False

    def is_model_present(self) -> bool:
        """Check if local model directory or weights exist at model_path."""
        p = Path(self.model_path)
        if not p.exists():
            return False
        if p.is_file():
            return True
        # Check if directory has model files
        for ext in ("*.onnx", "*.safetensors", "*.bin", "config.json"):
            if list(p.glob(ext)):
                return True
        return False

    def initialize(self, backend_type: Optional[str] = None) -> bool:
        """Initialize and bind inference backend."""
        self.backend = get_backend(backend_type)
        self.is_ready = self.backend.load_model(self.model_path)
        return self.is_ready

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2
    ) -> str:
        """Run text generation using loaded backend."""
        if not self.is_ready or self.backend is None:
            self.initialize()
        return self.backend.generate(prompt, max_new_tokens, temperature)

    def get_status(self) -> Dict[str, Any]:
        """Return runtime model and backend telemetry."""
        device = get_device_info()
        backend_info = self.backend.get_telemetry() if self.backend else {}

        return {
            "model_path": self.model_path,
            "model_present": self.is_model_present(),
            "model_ready": self.is_ready,
            "backend": backend_info.get("backend", device["backend"]),
            "accelerator": backend_info.get("accelerator", device["accelerator"]),
            "fallback_occurred": backend_info.get("fallback_occurred", False),
            "fallback_reason": backend_info.get("fallback_reason", None),
            "active_provider": backend_info.get("active_provider", "CPUExecutionProvider"),
        }


# Global singleton instance
model_manager = ModelManager()
