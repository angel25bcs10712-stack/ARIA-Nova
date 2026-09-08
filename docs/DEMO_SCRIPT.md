# ARIA Nova — 3-Minute Competition Pitch & Demo Script

**Competition**: Qualcomm Snapdragon AI Lab Build & Present Challenge  
**Project**: ARIA Nova — Adaptive On-Device Enterprise AI for Snapdragon PCs  
**Presenter**: Angel Singh  
**Total Duration**: 3 Minutes (180 Seconds)

---

## Pitch Timeline & Speaker Transcript

### ⏱️ 0:00 — The Problem: Static AI Breaks in Dynamic Enterprises (20s)
> *"Judges, enterprise knowledge workers switch between five or more applications every day. Today's LLM agents are trained in static environments where rules never change. But in real enterprises, the world doesn't stand still: flight policies get updated, budgets freeze, and approval rules change mid-flight. When rules change, standard AI agents blindly push ahead with outdated plans, causing compliance disasters and security violations."*

---

### ⏱️ 0:20 — Introduce ARIA Nova (20s)
> *"Introducing **ARIA Nova**: an adaptive on-device enterprise AI agent built on the ARIA OpenEnv workspace. ARIA Nova executes long-horizon workflows while dynamically detecting and adapting to mid-session enterprise policy changes. Even better: it brings this capability completely on-device, designed specifically for Snapdragon-powered Windows PCs."*

---

### ⏱️ 0:40 — Hardware Telemetry & Runtime Transparency (20s)
*(Screen displays terminal or Gradio UI banner)*
> *"Let's examine the live system. Running `python -m edge.device`, notice our hardware detection is completely truthful: on this development workstation, it truthfully reports `Backend: CPU` with `CPUExecutionProvider` fallback. When deployed onto a Snapdragon X Elite PC, it engages the Qualcomm Neural Network (QNN) Execution Provider and Hexagon NPU. No fake benchmarks, no synthetic claims."*

---

### ⏱️ 1:00 — Starting the Enterprise Workflow (30s)
*(Presenter runs `python demo.py`)*
> *"Our task today: 'Arrange an international business meeting for an employee according to company policy.'*
> *Under corporate Policy v1, international travel is authorized with standard direct manager approval and a $350 per diem limit.*
> *Watch ARIA Nova begin the workflow: it checks the employee's calendar, reserves the Thursday 2 PM slot for a London client briefing, and formats an email to the direct manager."*

---

### ⏱️ 1:30 — Injecting Policy Drift Mid-Execution (20s)
> *"Now, at Step 4, organizational policy drifts! Human resources and finance issue Policy Directive v2: due to budget consolidation, all international travel now strictly requires VICE PRESIDENT approval, and per diem drops to $250."*

---

### ⏱️ 1:50 — Instant Policy Change Detection (10s)
> *"Before committing the reservation, ARIA Nova's policy engine inspects active rules. The agent detects the version change immediately: `POLICY CHANGE DETECTED DURING EXECUTION`."*

---

### ⏱️ 2:00 — Dynamic Re-Planning & Adaptation (20s)
> *"Instead of blindly executing the old manager approval, ARIA Nova halts dispatch, dynamically re-plans its workflow, and escalates authorization directly to the Vice President at `vp@corp.com`. It logs the expense in the spreadsheet ledger strictly within the new $250 limit. The task succeeds with a 100% adaptation score!"*

---

### ⏱️ 2:20 — Air-Gapped Offline Mode & Privacy (20s)
> *"All of this executed with `ARIA_OFFLINE_MODE=true`. Zero employee emails, meeting times, or financial records were transmitted to external cloud APIs. Enterprise data stays on the device, ensuring 100% GDPR and SOC 2 compliance."*

---

### ⏱️ 2:40 — Benchmarking & Verified Performance (15s)
*(Show `python -m edge.benchmark` output)*
> *"Our integrated benchmark profiles model load time, per-token latency, and memory footprint. Everything runs locally in under 250 milliseconds total workflow latency, utilizing less than 60 MB of process memory."*

---

### ⏱️ 2:55 — Snapdragon Deployment Path (5s)
> *"With our ONNX Runtime QNN backend and Qualcomm AI Hub optimization pathway, ARIA Nova is ready to leverage Snapdragon X Elite's 45 TOPS Hexagon NPU for unprecedented efficiency."*

---

### ⏱️ 3:00 — Conclusion & Q&A
> *"ARIA Nova: Adaptive enterprise intelligence, local privacy, powered by Qualcomm Snapdragon. Thank you, and I welcome your questions!"*
