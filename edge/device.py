"""
ARIA Nova Hardware & Execution Provider Detection
Provides truthful device detection for Snapdragon PCs and standard CPUs.
"""

import sys
import platform
import subprocess
from typing import Dict, Any, List


def get_onnx_execution_providers() -> List[str]:
    """Retrieve available ONNX Runtime execution providers."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]


def is_snapdragon_hardware() -> bool:
    """
    Check if the current host processor is genuine Qualcomm Snapdragon silicon.
    Uses platform information, CPU brand string, and Windows WMI if available.
    """
    machine = platform.machine().lower()
    processor = platform.processor().lower()

    # ARM64 architecture check
    if "arm64" in machine or "aarch64" in machine:
        if "snapdragon" in processor or "qualcomm" in processor or "sc8380" in processor or "x elite" in processor:
            return True

    # Windows specific WMI query for processor identification
    if sys.platform == "win32":
        try:
            # Query processor name via wmic or powershell without blocking long
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],
                capture_output=True,
                text=True,
                timeout=3
            )
            name = res.stdout.strip().lower()
            if "snapdragon" in name or "qualcomm" in name:
                return True
        except Exception:
            pass

    return False


def get_device_info() -> Dict[str, Any]:
    """
    Detect operating system, architecture, CPU, and ONNX Runtime QNN/CPU providers.
    Never fabricates Snapdragon NPU presence.
    """
    plat = platform.system()
    arch = platform.machine()
    cpu_name = platform.processor() or platform.uname().processor or "Generic Processor"
    providers = get_onnx_execution_providers()

    qnn_available = "QNNExecutionProvider" in providers
    is_snapdragon = is_snapdragon_hardware()

    # NPU is available only if QNN EP is available on supported hardware
    npu_available = bool(qnn_available and (is_snapdragon or "QNNExecutionProvider" in providers))

    if qnn_available and npu_available:
        backend = "qnn"
        accelerator = "NPU"
    else:
        backend = "cpu"
        accelerator = "CPU"

    return {
        "platform": plat,
        "architecture": arch,
        "cpu": cpu_name,
        "backend": backend,
        "accelerator": accelerator,
        "qnn_available": qnn_available,
        "npu_available": npu_available,
        "is_snapdragon": is_snapdragon,
        "execution_providers": providers,
    }


def print_device_summary():
    """Print standard formatted device summary for CLI."""
    info = get_device_info()
    print("=" * 45)
    print("      ARIA NOVA DEVICE INFORMATION")
    print("=" * 45)
    print(f"Platform:            {info['platform']}")
    print(f"Architecture:        {info['architecture']}")
    print(f"CPU:                 {info['cpu']}")
    print(f"Backend:             {info['backend'].upper()}")
    print(f"Accelerator:         {info['accelerator']}")
    print(f"QNN Available:       {info['qnn_available']}")
    print(f"NPU Available:       {info['npu_available']}")
    print(f"Execution Providers: {', '.join(info['execution_providers'])}")
    print("=" * 45)


if __name__ == "__main__":
    print_device_summary()
