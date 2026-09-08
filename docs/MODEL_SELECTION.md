# ARIA Nova — Model Selection & Edge Deployment Analysis

This document evaluates Small Language Model (SLM) candidates for local deployment on Snapdragon PCs and CPU workstations.

---

## 1. Candidate Comparison Matrix

| Model Architecture | Parameters | FP16 Size | INT4 Size | Tokenizer | Native ONNX Support | AI Hub Target Profile | Snapdragon NPU Feasibility |
|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| **Qwen2.5-1.5B-Instruct** | 1.54B | ~3.1 GB | ~950 MB | BPE (151k vocab) | ✅ Yes (Optimum) | `Snapdragon X Elite CRD` | ⭐⭐⭐⭐⭐ Highest (Recommended) |
| **Llama-3.2-1B-Instruct** | 1.23B | ~2.5 GB | ~780 MB | Tiktoken (128k vocab) | ✅ Yes (Optimum) | `Snapdragon X Elite CRD` | ⭐⭐⭐⭐⭐ Ultra-fast inference |
| **Phi-3.5-mini-instruct** | 3.82B | ~7.6 GB | ~2.3 GB | SentencePiece (32k vocab) | ✅ Yes (DirectML / ONNX) | `Snapdragon X Elite` | ⭐⭐⭐⭐ High quality reasoning |
| **Qwen2.5-7B-Instruct** | 7.61B | ~15.2 GB | ~4.5 GB | BPE (151k vocab) | ✅ Yes (Optimum) | Hybrid CPU/NPU | ⭐⭐⭐ Memory bound on 16GB PCs |

---

## 2. Selected Primary SLM: Qwen2.5-1.5B-Instruct

### Why Qwen2.5-1.5B?
1. **Instruction Following Fidelity**: Outstanding JSON and structured schema adherence (`TOOL`, `OPERATION`, `PARAMS`), critical for deterministic tool execution in OpenEnv.
2. **Compact INT4 Footprint**: Compresses to under 1 GB, fitting easily into the 45 TOPS Hexagon NPU memory workspace alongside Windows 11 system tasks.
3. **Curriculum Continuity**: Matches the foundational model architecture utilized during ARIA's original GRPO reinforcement learning training.

---

## 3. Local Inference Strategies

### Strategy 1: ONNX Runtime with QNN EP (Snapdragon Target)
- Graph compiled with INT4/W4A16 weight quantization.
- Runtime delegates linear layers and attention projections to `QnnHtp.dll` (Hexagon Tensor Processor).
- CPU handles token decoding orchestration.

### Strategy 2: CPU Fallback Engine (Development Target)
- Standard PyTorch or ONNX Runtime `CPUExecutionProvider` graph execution.
- If model weights are not downloaded, ARIA Nova's integrated deterministic rule-adaptation engine activates automatically, guaranteeing 100% test passing and offline reproducibility.

---

## 4. Qualcomm AI Hub Optimization Status

- **Architecture Support**: Both Qwen2.5 and Llama-3.2 architectures are supported by Qualcomm AI Hub compilation tools (`qai-hub`).
- **Verification Rule**: We do not claim completed AI Hub cloud profiling jobs until an active Qualcomm AI Hub API key and physical Snapdragon hardware have verified the compiled artifact.
