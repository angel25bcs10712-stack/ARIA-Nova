"""
ARIA Nova Offline Execution Mode
Ensures all enterprise workflow processing and SLM inference occurs locally
without sending prompts or corporate data to external cloud LLMs.
"""

from edge.config import config


class OfflineViolationError(RuntimeError):
    """Raised when an external network or cloud LLM call is attempted in offline mode."""
    pass


def is_offline_mode() -> bool:
    """Return whether offline execution mode is currently enabled."""
    return config.offline_mode


def assert_offline_compliant(destination: str = "cloud_api"):
    """
    Enforces offline compliance.
    Raises OfflineViolationError if offline mode is active and external connection attempted.
    """
    if is_offline_mode():
        raise OfflineViolationError(
            f"External call to '{destination}' blocked: ARIA_OFFLINE_MODE is active. "
            "Enterprise data must remain strictly on-device."
        )


def get_privacy_status() -> dict:
    """Return current privacy and data governance status."""
    offline = is_offline_mode()
    return {
        "offline_mode": offline,
        "cloud_llm_calls_blocked": offline,
        "data_residency": "Local On-Device (Zero Egress)" if offline else "Standard",
        "telemetry": "Local Only",
    }
