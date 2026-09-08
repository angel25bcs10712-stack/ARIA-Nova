"""
ARIA Nova Health and System Diagnostics
Monitors on-device runtime status, memory consumption, and backend readiness.
"""

import psutil
from typing import Dict, Any
from edge.config import config
from edge.device import get_device_info
from edge.model_manager import model_manager
from edge.offline import is_offline_mode


def get_health_status() -> Dict[str, Any]:
    """Retrieve full system health report."""
    device = get_device_info()
    model_status = model_manager.get_status()
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        "status": "ok",
        "model_loaded": model_status["model_ready"],
        "backend": model_status["backend"],
        "accelerator": model_status["accelerator"],
        "offline_mode": is_offline_mode(),
        "device": {
            "platform": device["platform"],
            "architecture": device["architecture"],
            "qnn_available": device["qnn_available"],
            "npu_available": device["npu_available"],
        },
        "memory": {
            "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
            "system_percent": psutil.virtual_memory().percent,
        },
    }
