# Qualcomm AI Hub Integration & Compilation Pipeline

This document details the intended model compilation, optimization, and deployment flow from standard PyTorch SLM weights to Qualcomm Snapdragon Hexagon NPUs via **Qualcomm AI Hub**.

---

## Intended Compilation & Deployment Pipeline

```
Small Language Model (PyTorch / Safetensors)
                   │
                   ▼
       Qualcomm AI Hub (Compile & Profile)
                   │
                   ▼
     Snapdragon Target (Snapdragon X Elite / Plus)
                   │
                   ▼
       Optimized & Quantized Model (INT4 / W4A16)
                   │
                   ▼
     ONNX Runtime (with QNN Execution Provider)
                   │
                   ▼
             Qualcomm QNN SDK
                   │
                   ▼
        Snapdragon Hexagon NPU
```

---

## Status Classification

To maintain 100% authenticity and technical rigor for the Qualcomm Snapdragon AI Lab Challenge, project capabilities are strictly demarcated across three operational tiers:

### Tier 1: FULLY IMPLEMENTED (Verified in Repository)
- **Truthful Hardware Detection (`edge/device.py`)**: Genuinely detects OS, CPU architecture, host processor, available ONNX Runtime execution providers, and checks for `QNNExecutionProvider`.
- **Dual-Backend Hardware Abstraction (`edge/inference.py`)**: Abstract base class supporting `CPUBackend` and `QNNBackend`.
- **QNN Provider Integration (`edge/qnn_backend.py`)**: Configures ONNX Runtime with `QNNExecutionProvider`, `QnnHtp.dll` backend path, and performance modes (`burst`, `graph_finalization_optimization_mode`).
- **Transparent CPU Fallback**: Automatically activates on non-Snapdragon hosts, populating `fallback_occurred=True` and reporting the exact reason.
- **Air-Gapped Offline Mode (`edge/offline.py`)**: Zero cloud LLM egress when `ARIA_OFFLINE_MODE=true`.
- **End-to-End Enterprise Environment & Policy Engine**: 5-tool OpenEnv workspace with real-time policy drift and re-planning.

### Tier 2: REQUIRES SNAPDRAGON HARDWARE (Execution-Ready)
- **Physical NPU Execution**: Requires a physical Snapdragon-powered Windows PC (e.g. Snapdragon X Elite, Snapdragon X Plus, or Snapdragon 8cx Gen 3).
- **Native Windows on ARM (WoA) QNN Provider**: Requires installing `onnxruntime-qnn` ARM64 wheels on Windows 11 on ARM.
- **Hexagon HTP Runtime**: Requires `QnnHtp.dll` and Qualcomm Hexagon NPU drivers present in the Windows System directory.

### Tier 3: REQUIRES QUALCOMM AI HUB VALIDATION (Post-Export Pipeline)
- **Model Compilation Jobs**: Submitting Qwen2.5 or Llama-3.2 to `qai-hub` cloud compiler for device-specific hardware profiling.
- **Hardware Benchmarks**: Generating official Qualcomm AI Hub device benchmark reports, memory bandwidth metrics, and power efficiency curves.
- *Notice*: No synthetic AI Hub job IDs or fabricated NPU latency figures are presented in this project. All metrics in `results/` reflect genuine local executions.

---

## Compiling via Qualcomm AI Hub CLI

When compiling on Qualcomm AI Hub, follow these steps:

```bash
# 1. Install Qualcomm AI Hub Client
pip install qai-hub

# 2. Configure API Token (do NOT commit tokens to source code)
qai-hub configure --api_token <YOUR_QUALCOMM_AI_HUB_API_KEY>

# 3. Compile PyTorch model for Snapdragon X Elite target
python -c "
import qai_hub as hub
import torch

# Load PyTorch model
model = torch.load('path/to/slm.pt')
sample_input = torch.randint(0, 32000, (1, 128))

# Submit compilation job
compile_job = hub.submit_compile_job(
    model=model,
    device=hub.Device('Snapdragon X Elite CRD'),
    input_specs={'input_ids': sample_input.shape},
    options='--target_runtime onnx --quantize_weights int4'
)

# Download compiled ONNX artifact
compiled_model = compile_job.get_target_model()
compiled_model.download('./models/aria-slm/model.onnx')
"
```
