"""Consolidate all rounds into the paper's active-learning CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from .config import DEFAULT_METHODS
from .strategies.cdu import VARIANTS


# Exact column order of the consolidated active-learning result.
PAPER_FIELDS = (
    "method", "iteration", "precision", "recall", "mAP50", "mAP50-95",
    "model_saved", "fitness", "class", "instances", "cl_p", "cl_r",
    "cl_ap50", "cl_ap", "test_precision", "test_recall", "test_mAP50",
    "test_mAP50-95", "test_fitness", "test_results_saved", "test_instances",
    "test_cl_p", "test_cl_r", "test_cl_ap50", "test_cl_ap", "alpha",
    "infer", "clusters", "db_index", "mult_factor",
)
CLASS_NAMES = ("crack", "patch", "pothole")
TEST_OBJECTS = {"crack": 60, "patch": 250, "pothole": 77}
PAPER_METHOD_NAMES = {
    "random": "random",
    "sum": "sum",
    "avg": "avg",
    "dua": "DUA",
    "cdu_fm4_g12": "CDU_FM4G12",
    "cdu_crec": "CDU_creciente",
    "cdu_sl": "CDU_SL",
    "cdu_sl_optpen": "CDU_SL_OptPen",
    "cdu_sl_opt055": "CDU_SL_Opt055",
    "cdu_sl_opt065": "CDU_SL_Opt065",
    "cdu_fm3_sl": "CDU_FM3_SL",
    "cdu_fm4_sl": "CDU_FM4_SL",
    "cdu_fm4_sl_opt055": "CDU_FM4_SL_OPT055",
}


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    """Replace the CSV only after all rows have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in PAPER_FIELDS} for row in rows)
    temporary.replace(path)


def _portable_path(value: str, repository_root: Path) -> str:
    """Store paths inside the published CSV relative to the repository."""
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Result path is outside the repository: {path}") from error


def refresh_results(output_root: Path) -> Path:
    """Rebuild ``active_learning_results.csv`` after every round.

    Each internal ``metrics.csv`` has one row per training run. Here, every
    round is expanded into three rows to retain metrics and object counts for
    ``crack``, ``patch``, and ``pothole`` in the format used by figures and tables.
    """
    output_root = Path(output_root)
    repository_root = output_root.parent
    results_dir = output_root / "results"
    destination = results_dir / "active_learning_results.csv"
    active = output_root / "active_learning"
    collected_rounds: list[tuple[str, dict]] = []
    baseline = _read(active / "baseline" / "metrics.csv")
    if baseline:
        collected_rounds.append(("baseline", baseline[0]))
    for method in DEFAULT_METHODS:
        for row in _read(active / method / "metrics.csv"):
            if int(row["round"]) > 0:
                collected_rounds.append((method, row))

    paper_rows: list[dict] = []
    for internal_method, round_metrics in collected_rounds:
        is_baseline = internal_method == "baseline"
        method = "basic" if is_baseline else PAPER_METHOD_NAMES[internal_method]
        factor = VARIANTS[internal_method].factor if internal_method in VARIANTS else ""

        # Global metrics repeat across the three rows for a round. The ``cl_*``
        # and ``test_cl_*`` fields change with the represented class.
        for class_name in CLASS_NAMES:
            paper_rows.append({
                "method": method,
                "iteration": "" if is_baseline else int(round_metrics["round"]) - 1,
                "precision": round_metrics["precision"],
                "recall": round_metrics["recall"],
                "mAP50": round_metrics["map50"],
                "mAP50-95": round_metrics["map5095"],
                "model_saved": _portable_path(round_metrics["checkpoint"], repository_root),
                "fitness": round_metrics["fitness"],
                "class": class_name,
                "instances": round_metrics[f"{class_name}_objects"],
                "cl_p": round_metrics[f"{class_name}_precision"],
                "cl_r": round_metrics[f"{class_name}_recall"],
                "cl_ap50": round_metrics[f"{class_name}_ap50"],
                "cl_ap": round_metrics[f"{class_name}_ap"],
                "test_precision": round_metrics["test_precision"],
                "test_recall": round_metrics["test_recall"],
                "test_mAP50": round_metrics["test_map50"],
                "test_mAP50-95": round_metrics["test_map5095"],
                "test_fitness": round_metrics["test_fitness"],
                "test_results_saved": _portable_path(
                    round_metrics["test_results_saved"], repository_root
                ),
                "test_instances": TEST_OBJECTS[class_name],
                "test_cl_p": round_metrics[f"test_{class_name}_precision"],
                "test_cl_r": round_metrics[f"test_{class_name}_recall"],
                "test_cl_ap50": round_metrics[f"test_{class_name}_ap50"],
                "test_cl_ap": round_metrics[f"test_{class_name}_ap"],
                "alpha": round_metrics.get("alpha", ""),
                "infer": "" if is_baseline else 20,
                "clusters": round_metrics.get("clusters", ""),
                "db_index": round_metrics.get("db_index", ""),
                "mult_factor": factor,
            })

    _atomic_csv(destination, paper_rows)
    duplicate = results_dir / "metrics_all.csv"
    if duplicate.exists():
        duplicate.unlink()
    return destination


__all__ = ["refresh_results", "PAPER_FIELDS", "PAPER_METHOD_NAMES"]
