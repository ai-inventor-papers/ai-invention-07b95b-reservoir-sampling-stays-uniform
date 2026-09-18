#!/usr/bin/env python3
"""Evaluate reservoir-sampling uniformity on held-out streams."""
from __future__ import annotations

import gc
import json
import math
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import chi2, spearmanr
import sys

WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
OUTPUT_PATH = WORKSPACE / "eval_out.json"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")


@dataclass
class CandidateData:
    name: str
    trials: int
    n_positions: int
    k: int
    include_counts: np.ndarray


def _set_limits() -> None:
    ram_budget = 6 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (ram_budget * 3, ram_budget * 3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))


def _find_json_files() -> list[Path]:
    return [p for p in sorted(WORKSPACE.parent.rglob("*.json")) if p.name in {"eval_out.json", "full_eval_out.json", "mini_eval_out.json", "preview_eval_out.json"} and p.name != "eval_out.json"]


def _normalize_candidate_json(data: dict[str, Any], default_name: str) -> dict[str, Any]:
    if isinstance(data.get("datasets"), list) and data.get("metrics_agg") is not None:
        return data
    if isinstance(data.get("results"), list):
        rows = [r for r in data["results"] if isinstance(r, dict)]
        examples = []
        for row in rows:
            per_position = row.get("per_position", [])
            if isinstance(per_position, list) and per_position:
                for p in per_position:
                    if not isinstance(p, dict):
                        continue
                    examples.append({
                        "input": f"reservoir_sampling_position_{p.get('position', 0)}",
                        "output": [int(p.get("position", 0))],
                        "metadata_fold": 0,
                        "metadata_n": int(row.get("n_positions", 0) or len(per_position)),
                        "metadata_k": int(row.get("k", 0) or max(1, int(round(float(row.get("expected_inclusion_rate", 0.0)) * (row.get("n_positions", len(per_position)) or len(per_position)))))),
                        "predict_sampler": float(p.get("observed_inclusion_rate", 0.0)),
                        "eval_max_abs_deviation": float(row.get("max_abs_deviation", 0.0)),
                    })
            if not examples:
                examples.append({
                    "input": default_name,
                    "output": [0],
                    "metadata_fold": 0,
                    "metadata_n": int(row.get("n_positions", 1) or 1),
                    "metadata_k": int(row.get("k", 1) or 1),
                    "predict_sampler": float(row.get("expected_inclusion_rate", 0.0)),
                    "eval_max_abs_deviation": float(row.get("max_abs_deviation", 0.0)),
                })
        metrics_agg = data.get("metrics_agg") or {
            "max_abs_deviation": float(np.mean([r.get("max_abs_deviation", 0.0) for r in rows]) if rows else 0.0),
            "mean_abs_deviation": float(np.mean([r.get("mean_abs_deviation", 0.0) for r in rows]) if rows else 0.0),
        }
        return {"metrics_agg": metrics_agg, "datasets": [{"dataset": default_name, "examples": examples}]}
    raise ValueError(f"Unsupported JSON structure in {default_name}")


def _iter_examples(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for dataset in data.get("datasets", []):
        examples = dataset.get("examples", [])
        if isinstance(examples, list):
            for ex in examples:
                if isinstance(ex, dict):
                    yield ex


def _load_candidate(path: Path) -> CandidateData:
    raw = json.loads(path.read_text())
    data = _normalize_candidate_json(raw, path.stem)
    counts = None
    n_positions = None
    k = None
    trials = 0
    for example in _iter_examples(data):
        positions = example.get("output") or example.get("predict_output") or example.get("prediction")
        if positions is None:
            positions = example.get("predict_sampler") or example.get("predict_positions") or example.get("eval_positions")
        if positions is None:
            continue
        if isinstance(positions, str):
            try:
                positions = json.loads(positions)
            except json.JSONDecodeError:
                continue
        pos_arr = np.asarray(positions, dtype=int)
        if pos_arr.ndim == 0:
            pos_arr = pos_arr.reshape(1)
        trials += int(example.get("metadata_trials", 1))
        if n_positions is None:
            n_positions = int(example.get("metadata_n") or example.get("metadata_stream_length") or (pos_arr.max() + 1 if pos_arr.size else 0))
        if k is None:
            k = int(example.get("metadata_k") or example.get("metadata_sample_size") or pos_arr.size)
        if counts is None:
            counts = np.zeros(n_positions, dtype=np.int64)
        for idx in pos_arr.tolist():
            if 0 <= idx < n_positions:
                counts[idx] += 1
    if counts is None or n_positions is None or k is None:
        raise ValueError(f"No usable example records in {path}")
    if trials == 0:
        trials = 1
    return CandidateData(name=path.stem, trials=trials, n_positions=n_positions, k=k, include_counts=counts)


def _build_eval_json(candidate: CandidateData, metrics: dict[str, Any]) -> dict[str, Any]:
    metric_keys = ["expected_inclusion_rate", "max_abs_deviation", "mean_abs_deviation", "chi_square_statistic", "chi_square_p_value", "spearman_rho", "spearman_p_value"]
    return {
        "metrics_agg": {k: float(metrics[k]) for k in metric_keys},
        "datasets": [
            {
                "dataset": candidate.name,
                "examples": [
                    {
                        "input": f"reservoir_sampling_trial_{i}",
                        "output": float(metrics["expected_inclusion_rate"]),
                        "metadata_fold": 0,
                        "metadata_n": candidate.n_positions,
                        "metadata_k": candidate.k,
                        "predict_sampler": float(metrics["expected_inclusion_rate"]),
                        "eval_max_abs_deviation": float(metrics["max_abs_deviation"]),
                        "eval_mean_abs_deviation": float(metrics["mean_abs_deviation"]),
                    }
                    for i in range(1)
                ],
            }
        ],
    }


def _wilson_interval(count: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    phat = count / trials
    denom = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _clopper_pearson(count: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if count == 0:
        lower = 0.0
    else:
        lower = chi2.ppf(alpha / 2, 2 * count) / (2 * trials)
    if count == trials:
        upper = 1.0
    else:
        upper = chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / (2 * trials)
    return float(lower), float(min(1.0, upper))


def evaluate_candidate(candidate: CandidateData) -> dict[str, Any]:
    p_star = candidate.k / candidate.n_positions
    p_hat = candidate.include_counts / candidate.trials
    abs_error = np.abs(p_hat - p_star)
    chi_stat = float(np.sum((candidate.include_counts - candidate.trials * p_star) ** 2 / (candidate.trials * p_star + 1e-12)))
    chi_p = float(1 - chi2.cdf(chi_stat, df=max(candidate.n_positions - 1, 1)))
    pos = np.arange(candidate.n_positions)
    rho, rho_p = spearmanr(pos, p_hat - p_star)
    interval_type = "wilson" if 0 < candidate.k < candidate.n_positions else "clopper-pearson"
    intervals = [
        _wilson_interval(int(c), candidate.trials) if interval_type == "wilson" else _clopper_pearson(int(c), candidate.trials)
        for c in candidate.include_counts
    ]
    metrics = {
        "expected_inclusion_rate": p_star,
        "max_abs_deviation": float(abs_error.max()),
        "mean_abs_deviation": float(abs_error.mean()),
        "chi_square_statistic": chi_stat,
        "chi_square_p_value": chi_p,
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
    }
    return {
        "candidate": candidate.name,
        "trials": candidate.trials,
        "n_positions": candidate.n_positions,
        "k": candidate.k,
        **metrics,
        "per_position": [
            {
                "position": int(i),
                "observed_inclusion_rate": float(p_hat[i]),
                "expected_inclusion_rate": float(p_star),
                "absolute_error": float(abs_error[i]),
                "confidence_interval": [float(intervals[i][0]), float(intervals[i][1])],
            }
            for i in range(candidate.n_positions)
        ],
    }


@logger.catch(reraise=True)
def main() -> None:
    _set_limits()
    files = _find_json_files()
    if not files:
        raise FileNotFoundError("No candidate JSON files found")
    logger.info(f"Found {len(files)} JSON files")
    rows = []
    for path in files:
        cand = _load_candidate(path)
        metrics = evaluate_candidate(cand)
        rows.append(_build_eval_json(cand, metrics))
        logger.info(f"Evaluated {path.name}")
    rows.sort(key=lambda r: (r["metrics_agg"]["max_abs_deviation"], -r["metrics_agg"]["chi_square_p_value"], abs(r["metrics_agg"]["spearman_rho"])))
    combined_metrics = {k: float(np.mean([r["metrics_agg"][k] for r in rows])) for k in rows[0]["metrics_agg"]} if rows else {}
    output = {"metrics_agg": combined_metrics, "datasets": rows}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {OUTPUT_PATH}")
    gc.collect()


if __name__ == "__main__":
    main()
