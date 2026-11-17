
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON safely without leaving partially written reports."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write complete results and summary JSON reports."""

    directory = Path(output_dir)

    results_path = directory / "results.json"
    summary_path = directory / "summary.json"

    _atomic_write_json(results_path, report)

    _atomic_write_json(
        summary_path,
        {
            "dataset_name": report["dataset_name"],
            "dataset_version": report["dataset_version"],
            "collection": report["collection"],
            "embedding_model": report["embedding_model"],
            "k_values": report["k_values"],
            "summary": report["summary"],
        },
    )

    return results_path, summary_path
