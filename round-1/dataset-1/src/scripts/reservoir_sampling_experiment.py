#!/usr/bin/env python3
"""Generate and evaluate a compact reservoir-sampling benchmark."""

from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = WORKSPACE_DIR / "results"
LOGS_DIR = WORKSPACE_DIR / "logs"
TEMP_DATASETS_DIR = WORKSPACE_DIR / "temp" / "datasets"

for _dir in (RESULTS_DIR, LOGS_DIR, TEMP_DATASETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

N_VALUES = (32, 64, 128)
BENCHMARK_TRIALS_DEFAULT = 50
EVAL_TRIALS_DEFAULT = 10000
BASE_SEED_DEFAULT = 20260918


logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS_DIR / "run.log", rotation="30 MB", level="DEBUG")


def _detect_cpus() -> int:
    """Detect container CPU allocation using cgroup/affinity."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().strip().split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return 1


def _set_memory_limits() -> None:
    """Set conservative memory limits so failures are catchable."""
    ram_budget = 1 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (ram_budget * 3, ram_budget * 3))
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
    except (ValueError, OSError):
        pass


def build_settings() -> list[dict[str, int]]:
    """Build a compact grid covering small, medium, and near-extreme reservoirs."""
    settings: list[dict[str, int]] = []
    for n in N_VALUES:
        ks = sorted({1, max(2, n // 4), max(2, n // 2), n - 1})
        for k in ks:
            if 1 <= k < n:
                settings.append({"N": n, "k": k})
    return settings


def reservoir_sample_positions(n: int, k: int, seed: int) -> list[int]:
    """Run Algorithm R and return the selected stream positions (1-based)."""
    if not (1 <= k < n):
        raise ValueError(f"Expected 1 <= k < n, got n={n}, k={k}")

    rng = np.random.default_rng(seed)
    reservoir = list(range(1, k + 1))
    for position in range(k + 1, n + 1):
        slot = int(rng.integers(1, position + 1))
        if slot <= k:
            reservoir[slot - 1] = position
    return reservoir


def build_benchmark_records(settings: list[dict[str, int]], trials_per_setting: int, base_seed: int) -> list[dict[str, Any]]:
    """Create one record per stream item per trial."""
    records: list[dict[str, Any]] = []
    for setting_idx, setting in enumerate(settings):
        n = setting["N"]
        k = setting["k"]
        expected = k / n
        stream_id = f"N{n:03d}_K{k:03d}"
        for trial_id in range(trials_per_setting):
            seed = base_seed + setting_idx * 100_000 + trial_id
            for position_index in range(1, n + 1):
                records.append(
                    {
                        "stream_id": stream_id,
                        "position_index": position_index,
                        "item_id": f"item_{position_index:05d}",
                        "N": n,
                        "k": k,
                        "seed": seed,
                        "trial_id": trial_id,
                        "expected_inclusion_prob": expected,
                    }
                )
    return records


def _evaluate_setting(args: tuple[int, int, int, int, int]) -> dict[str, Any]:
    """Evaluate one setting in a worker process."""
    n, k, trials_per_setting, base_seed, setting_idx = args
    counts = np.zeros(n, dtype=np.int64)
    stream_id = f"N{n:03d}_K{k:03d}"
    setting_seed = base_seed + setting_idx * 100_000

    for trial_id in range(trials_per_setting):
        seed = setting_seed + trial_id
        reservoir = reservoir_sample_positions(n, k, seed)
        for position in reservoir:
            counts[position - 1] += 1

    expected = k / n
    freqs = counts / trials_per_setting
    deviations = np.abs(freqs - expected)
    per_position = [
        {
            "position_index": int(position),
            "empirical_frequency": float(freqs[position - 1]),
            "expected_frequency": float(expected),
            "absolute_deviation": float(deviations[position - 1]),
        }
        for position in range(1, n + 1)
    ]
    return {
        "stream_id": stream_id,
        "N": n,
        "k": k,
        "trials": trials_per_setting,
        "expected_inclusion_prob": expected,
        "max_abs_deviation": float(deviations.max()),
        "mean_abs_deviation": float(deviations.mean()),
        "per_position": per_position,
    }


def evaluate(settings: list[dict[str, int]], trials_per_setting: int, base_seed: int) -> dict[str, Any]:
    """Evaluate all settings using process-level parallelism."""
    num_workers = min(_detect_cpus(), len(settings))
    logger.info(f"Evaluating {len(settings)} settings with {trials_per_setting} trials each on {num_workers} worker(s)")
    start = time.perf_counter()
    tasks = [
        (setting["N"], setting["k"], trials_per_setting, base_seed, idx)
        for idx, setting in enumerate(settings)
    ]

    results: list[dict[str, Any]] = [None] * len(tasks)  # type: ignore[assignment]
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as pool:
        future_to_idx = {pool.submit(_evaluate_setting, task): idx for idx, task in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    elapsed = time.perf_counter() - start
    overall_max = max(result["max_abs_deviation"] for result in results)
    logger.info(f"Evaluation finished in {elapsed:.2f}s; overall max deviation={overall_max:.6f}")
    gc.collect()
    return {
        "num_trials_per_setting": trials_per_setting,
        "base_seed": base_seed,
        "elapsed_seconds": elapsed,
        "overall_max_abs_deviation": overall_max,
        "settings": results,
    }


def evaluate_exact(settings: list[dict[str, int]], trials_per_setting: int, base_seed: int) -> dict[str, Any]:
    """Backward-compatible alias for the main evaluation path."""
    return evaluate(settings=settings, trials_per_setting=trials_per_setting, base_seed=base_seed)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {path} ({path.stat().st_size} bytes)")


@logger.catch(reraise=True)
def main() -> None:
    _set_memory_limits()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=BENCHMARK_TRIALS_DEFAULT)
    parser.add_argument("--seed", type=int, default=BASE_SEED_DEFAULT)
    parser.add_argument("--eval-trials", type=int, default=EVAL_TRIALS_DEFAULT)
    parser.add_argument("--benchmark-path", type=Path, default=RESULTS_DIR / "reservoir_benchmark.json")
    parser.add_argument("--report-path", type=Path, default=RESULTS_DIR / "reservoir_sampling_report.json")
    args = parser.parse_args()

    settings = build_settings()
    logger.info(f"Built {len(settings)} reservoir settings")
    benchmark = build_benchmark_records(settings=settings, trials_per_setting=args.trials, base_seed=args.seed)
    write_json(args.benchmark_path, benchmark)

    report = evaluate(settings=settings, trials_per_setting=args.eval_trials, base_seed=args.seed)
    report["settings_grid"] = settings
    report["benchmark_path"] = str(args.benchmark_path)
    report["benchmark_trials_per_setting"] = args.trials
    write_json(args.report_path, report)


if __name__ == "__main__":
    main()
