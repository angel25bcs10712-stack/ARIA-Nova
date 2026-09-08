# ARIA Historical Training & Evaluation Results

> [!NOTE]
> This document preserves the original reinforcement learning (RL) training and evaluation records from the **Meta PyTorch OpenEnv Hackathon × Scaler 2026** (author: Angel Singh).
> These results document the multi-signal GRPO training run and curriculum progression on ARIA's 5-tool enterprise workspace.

---

## 1. Original ARIA Results (OpenEnv Hackathon 2026)

### Training Configuration
- **Algorithm**: Group Relative Policy Optimization (GRPO) via HuggingFace TRL
- **Optimization**: Unsloth
- **Base Model**: Qwen2.5-1.5B / Qwen2.5-7B-Instruct
- **Hardware**: Tesla T4 GPU (Google Colab)
- **Episodes**: 1,000 Steps across 3 Curriculum Stages

### Key Milestone Metrics

| Metric | Before Training (Baseline) | After GRPO Training | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Composite Reward Score** | 0.2700 | 0.3500 | +29.6% |
| **Peak Episode Reward** | - | 0.3500 (at Step 550) | - |
| **Enterprise Task Completion** | 24.0% | 78.0% | +54.0% absolute |
| **Policy Adaptation Rate** | 0.0% | 65.0% | +65.0% absolute |
| **Training Steps** | 0 | 1,000 | - |

---

## 2. Multi-Component Reward Formulation

The environment uses 4 independent reward components to prevent gaming and reward hacking:

$$R = 0.4 \times R_{\text{Task}} + 0.2 \times R_{\text{Efficiency}} + 0.2 \times R_{\text{Adaptation}} + 0.2 \times R_{\text{AntiHacking}}$$

- **Capped Mode** ($R \in [0, 1]$): Used in Stage 1 for stable policy gradient convergence.
- **Uncapped Mode** ($R \in [0, \infty)$): Used in Stages 2 & 3 to reward multi-step depth and speed.

---

## 3. 3-Stage Curriculum Progression

1. **Stage 1 (Static World, Capped Rewards)**:
   - Agent learns fundamental multi-app tool routing (Email, Calendar, Document, Spreadsheet).
2. **Stage 2 (Dynamic World, Uncapped Rewards)**:
   - Mid-session policy drift introduced at step 10. Agent learns to query `policy.get` upon rule changes.
3. **Stage 3 (Full Enterprise Complexity)**:
   - Competing deadlines, complex schedule conflicts, and dynamic policy updates.

---

## 4. Visual Evidence Artifacts

The following performance graphs were recorded during the original training run and remain available in `results/`:

- **Reward Progression**: `results/reward_curve.png`
- **Task Completion Convergence**: `results/task_completion.png`
- **Adaptation Score Trajectory**: `results/adaptation_score.png`

---

## 5. ARIA Nova Edge Results (Snapdragon AI Challenge)

In ARIA Nova, the trained policy capabilities are deployed to an on-device edge AI layer designed for Snapdragon PCs:
- **Measured Hardware Target**: Host CPU with transparent ONNX Runtime QNNExecutionProvider fallback.
- **Inference Latency**: Sub-millisecond local rule verification & single-digit millisecond local SLM response on local hardware.
- **Zero Cloud Egress**: 100% air-gapped compliance with zero API calls transmitted externally.
