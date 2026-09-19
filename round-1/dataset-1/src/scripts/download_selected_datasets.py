#!/usr/bin/env python3
"""Download two compact, well-documented benchmark datasets."""

from __future__ import annotations

import json
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger


WORKSPACE = Path(__file__).resolve().parent.parent
TEMP_DIR = WORKSPACE / "temp" / "datasets"
LOG_DIR = WORKSPACE / "logs"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "dataset_downloads.log", rotation="20 MB", level="DEBUG")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    config: str
    split: str
    out_prefix: str


SPECS = (
    DatasetSpec("fancyzhx/ag_news", "default", "train", "ag_news_train"),
    DatasetSpec("nyu-mll/glue", "sst2", "train", "glue_sst2_train"),
)


def _limit_memory() -> None:
    # Conservative ceiling for two small text datasets.
    ram_budget = 4 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (ram_budget * 3, ram_budget * 3))


def _save_dataset(dataset: Any, out_prefix: str) -> dict[str, Any]:
    parquet_path = TEMP_DIR / f"{out_prefix}.parquet"
    json_path = TEMP_DIR / f"{out_prefix}.json"

    # Save compact parquet for reuse, plus a tiny JSON preview for inspection.
    dataset.to_parquet(str(parquet_path))
    preview_df = dataset.select(range(min(3, len(dataset)))).to_pandas()
    preview_rows = preview_df.to_dict(orient="records")
    json_path.write_text(json.dumps(preview_rows, indent=2, ensure_ascii=False))

    return {
        "parquet": str(parquet_path),
        "json_preview": str(json_path),
        "num_rows": len(dataset),
        "size_bytes": parquet_path.stat().st_size,
        "columns": list(dataset.column_names),
    }


@logger.catch(reraise=True)
def main() -> None:
    _limit_memory()
    manifest: list[dict[str, Any]] = []
    for spec in SPECS:
        logger.info(f"Loading {spec.dataset_id} [{spec.config}/{spec.split}]")
        ds = load_dataset(spec.dataset_id, spec.config, split=spec.split)
        logger.info(f"Loaded {len(ds)} rows with columns {ds.column_names}")
        saved = _save_dataset(ds, spec.out_prefix)
        saved.update({"dataset_id": spec.dataset_id, "config": spec.config, "split": spec.split})
        manifest.append(saved)
        del ds
    manifest_path = TEMP_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
