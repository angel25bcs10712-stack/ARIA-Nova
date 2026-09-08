"""
ARIA Nova — Adaptive On-Device Enterprise AI for Snapdragon PCs
Edge AI inference and deployment layer with CPU fallback and QNN/NPU execution.
"""

from edge.config import config

__version__ = "2.0.0"

def get_device_info():
    from edge.device import get_device_info as _gdi
    return _gdi()

def get_backend(backend_type=None):
    from edge.inference import get_backend as _gb
    return _gb(backend_type)

def ARIANovaAgent(*args, **kwargs):
    from edge.agent import ARIANovaAgent as _ana
    return _ana(*args, **kwargs)

__all__ = ["config", "get_device_info", "get_backend", "ARIANovaAgent"]
