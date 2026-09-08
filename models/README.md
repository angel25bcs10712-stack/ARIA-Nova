# ARIA Nova — Local Small Language Model (SLM) Setup Guide

This directory hosts on-device model weights for **ARIA Nova: Adaptive On-Device Enterprise AI for Snapdragon PCs**.

> [!IMPORTANT]
> To comply with competition repository size guidelines and security best practices, raw model weight matrices (`.onnx`, `.safetensors`, `.bin`) are **not committed to the Git repository**. 
> When no weights are downloaded, ARIA Nova seamlessly initializes its built-in deterministic enterprise policy runtime, ensuring zero broken dependencies and full offline functionality out of the box.

---

## 1. Supported Model Families & Architecture

ARIA Nova is designed for Small Language Models (SLMs) ranging from 1B to 7B parameters:

| Model Candidate | Parameter Count | Context Window | Target Device Runtime | Target Speed (Est.) |
|:---|:---:|:---:|:---|:---:|
| **Qwen2.5-1.5B-Instruct** | 1.54B | 32k | Snapdragon NPU (INT4) / CPU | 45-65 tok/s (NPU) |
| **Llama-3.2-1B-Instruct** | 1.23B | 128k | Snapdragon NPU (INT4) / CPU | 55-80 tok/s (NPU) |
| **Phi-3.5-mini-instruct** | 3.82B | 128k | Snapdragon NPU / CPU | 25-40 tok/s (NPU) |
| **Qwen2.5-7B-Instruct** | 7.61B | 32k | Snapdragon X Elite CPU / Hybrid | 15-25 tok/s |

---

## 2. Directory Layout

Place downloaded or compiled models in `./models/aria-slm/`:

```
models/
├── README.md                 <-- This document
└── aria-slm/
    ├── model.onnx            <-- Exported ONNX computational graph
    ├── model.onnx.data       <-- Tensor weights (if >2GB)
    ├── tokenizer.json        <-- HuggingFace Fast Tokenizer
    ├── tokenizer_config.json <-- Tokenizer configurations
    └── config.json           <-- Model architecture metadata
```

Configure the location in `.env`:
```bash
ARIA_MODEL_PATH=./models/aria-slm
```

---

## 3. Installation & Preparation Methods

### Method A: Hugging Face SLM Download (CPU Development)
To download a lightweight model locally using `huggingface-cli` or Python:

```bash
# Download Qwen2.5-1.5B-Instruct to ./models/aria-slm
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer.save_pretrained('./models/aria-slm')
model.save_pretrained('./models/aria-slm')
"
```

### Method B: ONNX Model Export for ONNX Runtime & QNN
To convert the model to ONNX format compatible with ONNX Runtime:

```bash
python -m pip install optimum[onnxruntime]
optimum-cli export onnx --model Qwen/Qwen2.5-1.5B-Instruct --task text-generation-with-past ./models/aria-slm/
```

### Method C: Qualcomm AI Hub Compilation (Snapdragon NPU Target)
When deploying to a Snapdragon PC (e.g. Snapdragon X Elite / Plus):
1. Compile the PyTorch/ONNX graph on **Qualcomm AI Hub** targeting the `Snapdragon X Elite` SoC.
2. Quantize weights to `INT4 / W4A16` for maximum Hexagon NPU throughput and thermal efficiency.
3. Download the compiled model bundle and place the generated `.onnx` and contextual QNN runtime binaries in `./models/aria-slm/`.

---

## 4. Hardware Deployment Considerations

- **Snapdragon NPU (Hexagon)**: Requires INT4 or INT8 quantized ONNX graphs paired with `QNNExecutionProvider` and `QnnHtp.dll`.
- **CPU Fallback**: FP32 or FP16 graphs run universally across all x86_64 and ARM64 hosts via `CPUExecutionProvider`.
- **Zero Cloud Leakage**: When `ARIA_OFFLINE_MODE=true`, all weights remain completely resident on-device with zero internet socket egress.
