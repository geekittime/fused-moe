"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
flashinfer_cache_volume = modal.Volume.from_name("flashinfer-cache", create_if_missing=True)
TRACE_SET_PATH = "/data"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "PYTHONUNBUFFERED": "1"})
    .apt_install("ninja-build")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "apache-tvm-ffi")
)


def _patch_persistent_runner_health_check():
    """Avoid false health-check timeouts while keeping the official benchmark path."""
    from flashinfer_bench.bench.runner import persistent_runner

    def is_healthy(self):
        return (
            self._parent_conn is not None
            and self._worker_proc is not None
            and self._worker_proc.is_alive()
            and not self._parent_conn.closed
        )

    persistent_runner.PersistentSubprocessWorker.is_healthy = is_healthy


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={
        TRACE_SET_PATH: trace_volume,
        FLASHINFER_CACHE_PATH: flashinfer_cache_volume,
    },
)
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(
            warmup_runs=5,
            iterations=10,
            num_trials=1,
            timeout_seconds=900,
            use_isolated_runner=False,
        )

    import logging
    import os

    logging.disable(logging.INFO)
    logging.basicConfig(level=logging.WARNING, force=True)

    _patch_persistent_runner_health_check()

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    max_workloads = int(os.environ.get("FIB_MAX_WORKLOADS", "0"))
    if max_workloads > 0:
        workloads = workloads[:max_workloads]

    all_solutions = [solution]
    print(f"Testing {len(all_solutions)} solution(s): {[s.name for s in all_solutions]}")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: all_solutions},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    try:
        result_trace_set = benchmark.run_all(dump_traces=True)
    finally:
        benchmark.close()
        try:
            flashinfer_cache_volume.commit()
        except Exception:
            pass

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution if isinstance(trace.solution, str) else getattr(trace, "solution", ""),
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            sol_name = result.get("solution", "")
            print(f"  [{sol_name}] Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text(encoding="utf-8"))
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
