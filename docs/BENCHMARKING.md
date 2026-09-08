# ARIA Nova — Edge Benchmarking Methodology

ARIA Nova incorporates a rigorous, truthful hardware benchmarking suite located in `edge/benchmark.py` and `evaluation/edge_benchmark.py`.

---

## 1. Benchmarking Principles

1. **No Synthetic Benchmarks**: We never fabricate Snapdragon NPU latency numbers or token throughput when running on standard CPUs.
2. **Provider Transparency**: The exact execution provider reported by ONNX Runtime (`CPUExecutionProvider`, `QNNExecutionProvider`, etc.) is logged.
3. **Physical Memory Profiling**: Memory footprint is profiled before and after model allocation using OS-level `psutil.Process().memory_info()`.
4. **Reproducible Methodology**: Benchmarks include warmup runs, multi-iteration averages, and min/max latency bounds.

---

## 2. Running the Benchmarks

### Primary Edge Profiler:
```bash
python -m edge.benchmark
```
Measures model load latency, prompt processing latency, token throughput, and memory consumption. Results are saved to `results/edge_benchmark.json`.

### Hardware Comparison Suite:
```bash
python -m evaluation.edge_benchmark
```
Runs identical enterprise task prompts across available execution backends. Generates `results/edge_comparison.csv` and `results/latency_comparison.png` (if multiple backends are measured).

---

## 3. Metrics Specification

| Metric | Definition | Units | Tooling |
|:---|:---|:---:|:---|
| **Model Load Time** | Latency from disk load to memory ready state | ms | `time.perf_counter()` |
| **Inference Latency** | Time taken for generation across prompt + generation | ms | `time.perf_counter()` |
| **Throughput** | Output tokens generated per second | tokens/sec | `tokens / (latency_sec)` |
| **RSS Memory** | Resident Set Size of the host Python process | MB | `psutil.Process().memory_info().rss` |
| **Execution Provider** | Active provider utilized by ONNX Runtime | string | `ort.get_available_providers()` |
