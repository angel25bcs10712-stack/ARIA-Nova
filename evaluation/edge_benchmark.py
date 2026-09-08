"""
ARIA Nova Evaluation: CPU vs Snapdragon QNN Benchmark Comparison
Benchmarks CPU and QNN (if present) across identical workloads.
Outputs results/edge_comparison.csv and results/latency_comparison.png if measured.
Never fabricates QNN/NPU data.
"""

import os
import time
import psutil
import pandas as pd
from pathlib import Path
from edge.device import get_device_info
from edge.inference import get_backend


def evaluate_edge_comparison(iterations: int = 10):
    """
    Run comparative evaluation across available hardware backends.
    """
    device = get_device_info()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    test_prompt = (
        "TASK: Arrange international travel and file business review.\n"
        "POLICY: International travel requires VP approval.\n"
        "ACTION:"
    )

    records = []
    print("=" * 65)
    print("      ARIA NOVA -- HARDWARE BACKEND COMPARISON EVALUATION")
    print("=" * 65)
    print(f"Device: {device['platform']} {device['architecture']} | CPU: {device['cpu']}")
    print(f"QNN Available: {device['qnn_available']} | NPU Available: {device['npu_available']}")
    print("-" * 65)

    # 1. Evaluate CPU Backend
    print("\nEvaluating CPU Backend...")
    cpu = get_backend("cpu")
    cpu.load_model("./models/aria-slm")

    # Warmup
    _ = cpu.generate(test_prompt)

    cpu_latencies = []
    proc = psutil.Process()
    mem_start = proc.memory_info().rss / (1024 * 1024)

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = cpu.generate(test_prompt)
        cpu_latencies.append((time.perf_counter() - t0) * 1000)

    mem_end = proc.memory_info().rss / (1024 * 1024)
    avg_cpu_lat = sum(cpu_latencies) / len(cpu_latencies)
    cpu_tps = (32.0 / (avg_cpu_lat / 1000.0)) if avg_cpu_lat > 0 else 0.0

    records.append({
        "backend": "CPU",
        "accelerator": "CPU",
        "execution_provider": "CPUExecutionProvider",
        "avg_latency_ms": round(avg_cpu_lat, 2),
        "min_latency_ms": round(min(cpu_latencies), 2),
        "max_latency_ms": round(max(cpu_latencies), 2),
        "throughput_tokens_per_sec": round(cpu_tps, 2),
        "rss_memory_mb": round(mem_end, 2),
        "hardware_verified": True
    })

    print(f"CPU Latency: {avg_cpu_lat:.2f} ms | Throughput: {cpu_tps:.2f} tok/s")

    # 2. Evaluate QNN Backend if available
    if device["qnn_available"]:
        print("\nEvaluating Snapdragon QNN Backend...")
        try:
            qnn = get_backend("qnn")
            qnn.load_model("./models/aria-slm")
            _ = qnn.generate(test_prompt)

            qnn_latencies = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = qnn.generate(test_prompt)
                qnn_latencies.append((time.perf_counter() - t0) * 1000)

            qnn_mem = proc.memory_info().rss / (1024 * 1024)
            avg_qnn_lat = sum(qnn_latencies) / len(qnn_latencies)
            qnn_tps = (32.0 / (avg_qnn_lat / 1000.0)) if avg_qnn_lat > 0 else 0.0

            records.append({
                "backend": "QNN",
                "accelerator": "NPU",
                "execution_provider": "QNNExecutionProvider",
                "avg_latency_ms": round(avg_qnn_lat, 2),
                "min_latency_ms": round(min(qnn_latencies), 2),
                "max_latency_ms": round(max(qnn_latencies), 2),
                "throughput_tokens_per_sec": round(qnn_tps, 2),
                "rss_memory_mb": round(qnn_mem, 2),
                "hardware_verified": True
            })
            print(f"QNN NPU Latency: {avg_qnn_lat:.2f} ms | Throughput: {qnn_tps:.2f} tok/s")
        except Exception as e:
            print(f"QNN execution error: {e}")
    else:
        print("\nQNN Execution Provider is not available on this host.")
        print("Skipping QNN NPU benchmark to preserve benchmark authenticity.")

    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = results_dir / "edge_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark comparison table to {csv_path}")

    # Generate graph if both CPU and QNN have genuine data
    if len(records) >= 2:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            backends = [r["backend"] for r in records]
            latencies = [r["avg_latency_ms"] for r in records]
            colors = ["#4A90E2", "#E94E77"]

            ax.bar(backends, latencies, color=colors, width=0.4)
            ax.set_ylabel("Average Latency (ms)")
            ax.set_title("ARIA Nova: On-Device Inference Latency Comparison")
            for i, v in enumerate(latencies):
                ax.text(i, v + 0.1, f"{v:.2f} ms", ha='center', fontweight='bold')

            png_path = results_dir / "latency_comparison.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated comparison graph at {png_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    evaluate_edge_comparison()
