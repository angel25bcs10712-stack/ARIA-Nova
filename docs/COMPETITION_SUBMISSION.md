# Qualcomm Snapdragon AI Lab Build & Present Challenge — Submission Document

---

## Executive Summary

- **Project Name**: ARIA Nova
- **Tagline**: Adaptive On-Device Enterprise AI for Snapdragon PCs
- **One-Line Description**: An adaptive enterprise AI agent that executes long-horizon workflows locally while responding dynamically to real-time policy changes.
- **Author**: Angel Singh
- **Repository**: [https://github.com/angel25bcs10712-stack/ARIA-Nova](https://github.com/angel25bcs10712-stack/ARIA-Nova)

---

## 1. Problem Statement

Enterprise workflows are complex, multi-application processes involving emails, scheduling, document retrieval, and expense logging. Existing enterprise AI agents suffer from two fatal vulnerabilities:
1. **Static Assumption Failure**: Agents are trained in static environments and assume corporate rules never change mid-task. When travel rules, spending limits, or approval thresholds update mid-execution, agents proceed blindly with outdated plans, causing compliance failures.
2. **Cloud Egress & Privacy Risk**: Current agent frameworks transmit proprietary emails, corporate financial ledgers, and confidential schedules to external cloud LLMs, violating enterprise privacy, GDPR, and data residency laws.

---

## 2. Solution: ARIA Nova

ARIA Nova transforms the reinforcement learning capabilities of the ARIA OpenEnv enterprise environment into an **on-device, privacy-preserving edge agent designed specifically for Snapdragon-powered Windows PCs**.

By integrating an adaptive local AI agent with a hardware-aware backend abstraction, ARIA Nova executes enterprise workflows entirely on-device, detects mid-task policy drift instantly, and dynamically re-plans its actions without cloud dependencies.

---

## 3. Key Innovation

- **Mid-Session Policy Drift Adaptation**: Unlike static agents, ARIA Nova monitors policy signals across execution steps. If a corporate directive updates (e.g. escalating from direct manager approval to mandatory VP authorization), the agent intercepts execution, updates its belief state, and re-routes actions accordingly.
- **Hardware-Aware Backend Abstraction**: Seamlessly operates on Snapdragon PCs via ONNX Runtime with Qualcomm Neural Network (QNN) Execution Provider, while providing a verified CPU fallback for cross-platform development.
- **Air-Gapped Offline Execution**: Enforces `ARIA_OFFLINE_MODE=true` to ensure zero cloud data leakage.

---

## 4. Snapdragon Relevance & Hardware Alignment

- **Hexagon NPU Acceleration**: The architecture targets Qualcomm Hexagon NPUs (delivering up to 45 TOPS on Snapdragon X Elite) via ONNX Runtime `QNNExecutionProvider` and `QnnHtp.dll`.
- **Qualcomm AI Hub Alignment**: Follows the official Qualcomm AI Hub compilation path for quantization (INT4/W4A16) and device-targeted deployment.
- **Thermal & Battery Efficiency**: Offloading continuous enterprise policy monitoring and tool orchestration to the Snapdragon NPU enables all-day productivity without thermal throttling or battery drain.

---

## 5. Privacy & Data Governance

- **Zero Cloud API Dependencies**: Prompts, tool observations, and company data remain resident in physical device memory.
- **Air-Gapped Guardrails**: Active network guard (`edge/offline.py`) prevents unauthorized cloud LLM socket communication.

---

## 6. Technical Implementation Stack

```
User / UI Layer         -->  ARIA Nova Gradio Interface + FastAPI Edge Server
                                        │
Agent Layer             -->  Adaptive On-Device Agent (edge/agent.py)
                                        │
Model Management        -->  Model Manager (edge/model_manager.py)
                                        │
SLM Architecture        -->  Small Language Model (Qwen2.5-1.5B / Llama-3.2-1B)
                                        │
Hardware Backend        -->  Dual-Path Runtime (edge/inference.py)
                             ├─► Snapdragon QNN (ONNX Runtime + QNN EP + Hexagon NPU)
                             └─► Universal CPU Fallback (CPUExecutionProvider)
                                        │
Enterprise Simulation   -->  ARIA OpenEnv (Email, Calendar, Doc, Sheet, Policy Engine)
                                        │
Reward & Evaluation     -->  4-Component Reward Engine + Edge Benchmark Profiler
```

---

## 7. Authenticity & Verification Statement

In compliance with challenge ethics and technical integrity:
- Hardware detection (`edge/device.py`) truthfully queries host platform capabilities and ONNX Runtime execution providers.
- Snapdragon NPU acceleration is claimed **only when genuinely detected** on verified Snapdragon silicon.
- Standard workstations truthfully report CPU fallback. No synthetic benchmarks or fabricated AI Hub job IDs are used.
