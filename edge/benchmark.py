"""
ARIA Nova Edge Inference Benchmark
Measures actual on-device latency, model loading time, throughput, and memory footprint.
Saves measured values to results/edge_benchmark.json without fabricating NPU data.
"""

import os
import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any
from edge.device import get_device_info
from edge.inference import get_backend


def run_benchmark(num_iterations: int = 5) -> Dict[str, Any]:
    """
    Execute hardware benchmarking measuring real system and inference performance.
    """
    device = get_device_info()
    process = psutil.Process()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("        ARIA NOVA ON-DEVICE BENCHMARK")
    print("=" * 60)
    print(f"Platform:            {device['platform']}")
    print(f"Architecture:        {device['architecture']}")
    print(f"CPU:                 {device['cpu']}")
    print(f"Detected Backend:    {device['backend'].upper()}")
    print(f"Accelerator:         {device['accelerator']}")
    print(f"QNN Available:       {device['qnn_available']}")
    print(f"NPU Available:       {device['npu_available']}")
    print("-" * 60)

    # 1. Benchmark CPU Backend
    print("\n[1/2] Benchmarking CPU Inference Backend...")
    mem_before = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    cpu_backend = get_backend("cpu")
    cpu_backend.load_model("./models/aria-slm")
    cpu_load_time_ms = (time.perf_counter() - t0) * 1000

    test_prompt = "TASK: Schedule Q3 business review with client.\nTOOL: calendar\nOPERATION:"
    latencies = []

    # Warmup
    _ = cpu_backend.generate(test_prompt, max_new_tokens=32)

    for i in range(num_iterations):
        start = time.perf_counter()
        output = cpu_backend.generate(test_prompt, max_new_tokens=32)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    mem_after = process.memory_info().rss / (1024 * 1024)
    avg_latency_ms = sum(latencies) / len(latencies)
    est_tokens = 32
    tokens_per_sec = (est_tokens / (avg_latency_ms / 1000.0)) if avg_latency_ms > 0 else 0.0

    print(f"   Model Load Time:   {cpu_load_time_ms:.2f} ms")
    print(f"   Avg Latency:       {avg_latency_ms:.2f} ms ({num_iterations} runs)")
    print(f"   Throughput (est):  {tokens_per_sec:.2f} tokens/sec")
    print(f"   Memory Delta:      {max(0.0, mem_after - mem_before):.2f} MB (RSS: {mem_after:.2f} MB)")

    benchmark_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "cpu_benchmark": {
            "backend": "cpu",
            "execution_provider": "CPUExecutionProvider",
            "load_time_ms": round(cpu_load_time_ms, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "estimated_tokens_per_sec": round(tokens_per_sec, 2),
            "rss_memory_mb": round(mem_after, 2),
        },
        "qnn_benchmark": None
    }

    # 2. Benchmark QNN Backend if genuinely present
    print("\n[2/2] Checking Snapdragon QNN Execution Provider...")
    if not device["qnn_available"]:
        print("   QNN benchmark skipped:")
        print("   QNN Execution Provider is not available.")
        benchmark_data["qnn_benchmark"] = {
            "status": "skipped",
            "reason": "QNN Execution Provider is not available on host hardware."
        }
    else:
        print("   QNN Execution Provider detected! Running real NPU benchmark...")
        try:
            qnn_backend = get_backend("qnn")
            t_qnn0 = time.perf_counter()
            qnn_backend.load_model("./models/aria-slm")
            qnn_load_ms = (time.perf_counter() - t_qnn0) * 1000

            qnn_lats = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = qnn_backend.generate(test_prompt, max_new_tokens=32)
                qnn_lats.append((time.perf_counter() - start) * 1000)

            avg_qnn_ms = sum(qnn_lats) / len(qnn_lats)
            qnn_tps = (est_tokens / (avg_qnn_ms / 1000.0)) if avg_qnn_ms > 0 else 0.0

            benchmark_data["qnn_benchmark"] = {
                "status": "completed",
                "backend": "qnn",
                "execution_provider": "QNNExecutionProvider",
                "load_time_ms": round(qnn_load_ms, 2),
                "avg_latency_ms": round(avg_qnn_ms, 2),
                "estimated_tokens_per_sec": round(qnn_tps, 2),
            }
            print(f"   QNN Avg Latency:   {avg_qnn_ms:.2f} ms")
            print(f"   QNN Throughput:    {qnn_tps:.2f} tokens/sec")
        except Exception as e:
            benchmark_data["qnn_benchmark"] = {
                "status": "error",
                "reason": str(e)
            }
            print(f"   QNN benchmark error: {e}")

    # Save to results/edge_benchmark.json
    output_path = results_dir / "edge_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Benchmark results successfully saved to: {output_path}")
    print("=" * 60)

    return benchmark_data


if __name__ == "__main__":
    run_benchmark()
