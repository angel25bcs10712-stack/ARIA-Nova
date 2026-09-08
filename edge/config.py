"""
ARIA Nova Configuration Module
Handles environment variables and system settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Attempt to load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class EdgeConfig:
    """Central configuration for ARIA Nova edge runtime."""
    model_path: str = os.getenv("ARIA_MODEL_PATH", "./models/aria-slm")
    backend_mode: str = os.getenv("ARIA_BACKEND", "auto").lower()
    offline_mode: bool = os.getenv("ARIA_OFFLINE_MODE", "true").lower() in ("true", "1", "yes")
    log_level: str = os.getenv("ARIA_LOG_LEVEL", "INFO").upper()
    device_override: str = os.getenv("ARIA_DEVICE_OVERRIDE", "auto").lower()

    @property
    def model_dir(self) -> Path:
        return Path(self.model_path)


config = EdgeConfig()
