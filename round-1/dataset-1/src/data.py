#!/usr/bin/env python3
"""Generate a synthetic reservoir-sampling benchmark dataset."""
from __future__ import annotations

import json
import resource
from pathlib import Path

from loguru import logger
import sys

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "full_data_out.json"
DATA_DIR = ROOT / "temp" / "datasets"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "run.log", rotation="30 MB", level="DEBUG")


def _set_limits() -> None:
    budget = 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (budget * 3, budget * 3))


def _make_examples() -> list[dict[str, object]]:
    n_values = [16, 32, 64, 128]
    k_values = [1, 4, 8, 16]
    seeds = list(range(20))
    examples: list[dict[str, object]] = []
    for n in n_values:
        for k in k_values:
            if k > n:
                continue
            for seed in seeds:
                for pos in range(n):
                    examples.append(
                        {
                            "input": json.dumps(
                                {
                                    "stream_id": f"stream_N{n}_k{k}",
                                    "position_index": pos,
                                    "item_id": f"item_{pos}",
                                    "N": n,
                                    "k": k,
                                    "seed": seed,
                                    "trial_id": seed,
                                    "expected_inclusion_prob": k / n,
                                },
                                sort_keys=True,
                            ),
                            "output": str(k / n),
                            "metadata_row_index": pos,
                            "metadata_task_type": "regression",
                            "metadata_n_classes": 0,
                            "metadata_feature_names": [
                                "stream_id",
                                "position_index",
                                "item_id",
                                "N",
                                "k",
                                "seed",
                                "trial_id",
                                "expected_inclusion_prob",
                            ],
                            "metadata_dataset_name": "reservoir_sampling_benchmark",
                        }
                    )
    return examples


@logger.catch(reraise=True)
def main() -> None:
    _set_limits()
    logger.info("Generating synthetic benchmark rows")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    examples = _make_examples()
    payload = {"datasets": [{"dataset": "reservoir_sampling_benchmark", "examples": examples}]}
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {OUT_PATH} with {len(examples)} examples")


if __name__ == "__main__":
    main()
