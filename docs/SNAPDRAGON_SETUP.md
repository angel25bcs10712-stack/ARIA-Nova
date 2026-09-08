# Snapdragon PC Setup & Deployment Guide

This guide describes how to configure and run **ARIA Nova** natively on Qualcomm Snapdragon-powered Windows PCs (e.g. Snapdragon X Elite, Snapdragon X Plus, Surface Pro 11, Lenovo Yoga Slim 7x).

---

## 1. System Prerequisites

- **Operating System**: Windows 11 on ARM (Build 22631 or later)
- **Processor**: Qualcomm Snapdragon X Elite / Snapdragon X Plus / Snapdragon 8cx Gen 3
- **NPU Capability**: Qualcomm Hexagon NPU (up to 45 TOPS on Snapdragon X Elite)
- **Python**: Python 3.10, 3.11, or 3.12 (Native Windows ARM64 build recommended)
- **Qualcomm Drivers**: Latest Qualcomm Adreno GPU & Hexagon NPU drivers via Windows Update or OEM support page

---

## 2. Environment Installation

Clone the repository and install the Snapdragon-tailored dependencies:

```powershell
# 1. Clone repository
git clone https://github.com/angel25bcs10712-stack/ARIA-Nova.git
cd ARIA-Nova

# 2. Create Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install Snapdragon dependencies
pip install -r requirements-snapdragon.txt
```

---

## 3. Installing ONNX Runtime with QNN Execution Provider

To enable native Hexagon NPU offloading:

```powershell
# Install ONNX Runtime with Qualcomm Neural Network (QNN) execution provider
pip install onnxruntime-qnn --extra-index-url https://aihub.qualcomm.com/
```

Verify that the QNN execution provider is active:

```powershell
python -m edge.device
```

On a properly configured Snapdragon PC, the command will output:
```
=============================================
      ARIA NOVA DEVICE INFORMATION
=============================================
Platform:            Windows
Architecture:        ARM64
CPU:                 Snapdragon(R) X Elite - X1E80100
Backend:             QNN
Accelerator:         NPU
QNN Available:       True
NPU Available:       True
Execution Providers: QNNExecutionProvider, CPUExecutionProvider
=============================================
```

---

## 4. Running ARIA Nova on Snapdragon Hardware

### Run the Interactive Competition Demo:
```powershell
python demo.py
```

### Run the Hardware Benchmark:
```powershell
python -m edge.benchmark
```

### Launch the Gradio Web Interface:
```powershell
python app.py
```

### Launch the FastAPI Server:
```powershell
python server.py
```

---

## 5. Troubleshooting & Fallback Diagnostics

- **Fallback to CPU**: If `QNNExecutionProvider` is missing from the installed Python environment, ARIA Nova automatically logs:
  `QNNExecutionProvider not found in ONNX Runtime available providers. Falling back to CPUExecutionProvider.`
  and continues execution on the Snapdragon CPU without crashing.
- **Docker Usage Note**: For native NPU hardware acceleration, run ARIA Nova natively on the host Windows on ARM environment. Docker containers on Windows run inside a Linux VM that may not expose DirectML or Qualcomm QNN NPU pass-through without specialized hypervisor configurations.
