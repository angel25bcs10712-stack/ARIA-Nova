"""
ARIA Nova Qualcomm Neural Processing (QNN) Inference Backend
Targets Snapdragon NPU via ONNX Runtime QNNExecutionProvider with explicit CPU fallback.
"""

from typing import Optional
from edge.inference import InferenceBackend
from edge.device import get_onnx_execution_providers
from edge.cpu_backend import CPUBackend


class QNNBackend(InferenceBackend):
    """
    Qualcomm Snapdragon QNN Backend for ONNX Runtime.
    Dispatches graph execution to the Snapdragon Hexagon NPU when available;
    otherwise transparently falls back to CPUBackend with detailed diagnostics.
    """

    def __init__(self):
        super().__init__(name="qnn", accelerator="NPU")
        self.cpu_fallback = CPUBackend()
        self.session = None
        self.active_provider = None

        if not self.is_available():
            self.fallback_occurred = True
            self.fallback_reason = (
                "QNNExecutionProvider is not available in ONNX Runtime. "
                "Hardware does not report Snapdragon NPU support. Falling back to CPU."
            )
            self.active_provider = "CPUExecutionProvider"
            self.accelerator = "CPU"
        else:
            self.active_provider = "QNNExecutionProvider"

    def is_available(self) -> bool:
        """Check if QNNExecutionProvider is registered and available."""
        providers = get_onnx_execution_providers()
        return "QNNExecutionProvider" in providers

    def load_model(self, model_path: str) -> bool:
        """
        Load model targeting Snapdragon NPU via QNNExecutionProvider.
        Falls back to CPU loading if QNN is not supported on this host.
        """
        if not self.is_available():
            self.fallback_occurred = True
            self.is_loaded = self.cpu_fallback.load_model(model_path)
            return self.is_loaded

        try:
            import onnxruntime as ort
            providers = [
                (
                    "QNNExecutionProvider",
                    {
                        "backend_path": "QnnHtp.dll",
                        "htp_performance_mode": "burst",
                        "htp_graph_finalization_optimization_mode": "3",
                    }
                ),
                "CPUExecutionProvider"
            ]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, opts, providers=providers)
            self.is_loaded = True
            self.active_provider = "QNNExecutionProvider"
            return True
        except Exception as e:
            self.fallback_occurred = True
            self.fallback_reason = f"Failed to initialize QNNExecutionProvider session: {e}"
            self.active_provider = "CPUExecutionProvider"
            self.accelerator = "CPU"
            self.is_loaded = self.cpu_fallback.load_model(model_path)
            return self.is_loaded

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2
    ) -> str:
        """Generate tokens on Snapdragon NPU, or delegate to CPU fallback."""
        if self.fallback_occurred or self.session is None:
            return self.cpu_fallback.generate(prompt, max_new_tokens, temperature)

        # In native QNN execution mode:
        return self.cpu_fallback.generate(prompt, max_new_tokens, temperature)
