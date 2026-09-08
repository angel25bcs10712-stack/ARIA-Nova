# ARIA Nova — Air-Gapped Offline Execution Mode

In enterprise environments, proprietary financial models, internal meeting transcripts, executive travel schedules, and internal communications cannot be sent to third-party public cloud LLMs (such as OpenAI or Anthropic).

ARIA Nova solves this privacy vulnerability by providing a completely **air-gapped, on-device enterprise AI architecture**.

---

## 1. Why Offline On-Device AI Matters for Enterprise

1. **Zero Data Egress**: Enterprise tokens, employee PII, and financial ledgers never leave the physical RAM of the employee's Snapdragon laptop.
2. **Regulatory Compliance**: Meets stringent data residency requirements (GDPR, HIPAA, SOC 2 Type II, ISO 27001).
3. **Continuous Productivity**: Uninterrupted workflow execution in aircraft, secure defense facilities, and low-connectivity remote sites.
4. **Predictable Latency & Zero Cloud API Invoices**: Eliminates per-token cloud inference costs and network jitter.

---

## 2. Implementation & Enforcement Mechanism

Offline mode is governed by `edge/offline.py` and activated by setting:

```bash
ARIA_OFFLINE_MODE=true
```

When active:
- The agent loop strictly routes generation to local on-device SLMs (`edge/cpu_backend.py` or `edge/qnn_backend.py`).
- Any attempt by tools or models to establish outbound network connections to public cloud LLM endpoints raises an explicit `OfflineViolationError`.
- Telemetry reporting is constrained entirely to local JSON logs in `./results/`.

---

## 3. Verifying Offline Compliance

Verify that offline mode is active using the diagnostic command:

```bash
python -c "from edge.offline import get_privacy_status; print(get_privacy_status())"
```

Expected output:
```json
{
  "offline_mode": true,
  "cloud_llm_calls_blocked": true,
  "data_residency": "Local On-Device (Zero Egress)",
  "telemetry": "Local Only"
}
```
