"""Single serial entrypoint for the complete paper benchmark."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from active_learning_benchmark.active_learning import ActiveLearning
from active_learning_benchmark.config import DEFAULT_METHODS, FINAL_CDU_METHOD, METHODS, Config
from active_learning_benchmark.data import DatasetStore
from active_learning_benchmark.paper_plots import (
    execution_comparison, paper_figures, sensitivity_figures, update,
)


EXPECTED_RUNTIME = {
    "python": "3.10",
    "torch": "2.4.1+cu121",
    "ultralytics": "8.2.28",
    "numpy": "1.26.4",
    "scikit_learn": "1.5.1",
    "scipy": "1.13.1",
    "pandas": "2.2.2",
    "opencv": "4.10.0",
}

# Ultralytics 8.2.28 downloads this auxiliary model only to validate AMP.  It
# is not part of the experiment and must not remain beside the YOLOv9e source.
AMP_PROBE_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def _cleanup_amp_probe(root: Path) -> None:
    probe = root / "yolov8n.pt"
    if probe.is_file() and _sha256(probe) == AMP_PROBE_SHA256:
        probe.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime() -> dict[str, str | None]:
    import numpy
    import cv2
    import pandas
    import scipy
    import sklearn
    import torch
    import ultralytics
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "opencv": cv2.__version__,
        "nvidia_driver": driver,
    }


def _check_runtime(strict: bool) -> dict[str, str | None]:
    actual = _runtime()
    mismatches = []
    for key, expected in EXPECTED_RUNTIME.items():
        value = str(actual[key])
        matches = value.startswith(expected) if key == "python" else value == expected
        state = "OK" if matches else f"EXPECTED {expected}"
        print(f"runtime {key}: {value} [{state}]")
        if not matches:
            mismatches.append(f"{key}={value} (expected {expected})")
    print(f"runtime nvidia_driver: {actual['nvidia_driver']} (recorded, not pinned)")
    if strict and mismatches:
        raise RuntimeError(
            "The environment does not match the versions pinned for the benchmark: "
            + "; ".join(mismatches)
            + ". Recreate it from requirements.txt with Python 3.10."
        )
    return actual


def _configuration(args) -> Config:
    config_path = args.root.resolve() / "config.json"
    return Config.load(config_path).with_overrides(
        workers=args.workers,
        device=args.device,
        methods=tuple(args.methods),
    )


def _check(config: Config) -> None:
    weight = Path(config.model_source)
    if not weight.is_file():
        raise FileNotFoundError(f"Initial weight is missing: {weight}")
    digest = _sha256(weight)
    if digest != config.model_sha256:
        raise RuntimeError(
            f"Incorrect YOLOv9e weight: {digest}; expected {config.model_sha256}"
        )
    print(f"entrypoint: {Path(__file__).resolve()}")
    print(f"repository: {config.root}")
    print(f"model: {weight} sha256={digest}")
    final_weight = Path(config.final_model_source)
    if not final_weight.is_file():
        raise FileNotFoundError(f"Final CDU weight is missing: {final_weight}")
    final_digest = _sha256(final_weight)
    if final_digest != config.final_model_sha256:
        raise RuntimeError(
            f"Incorrect final CDU weight: {final_digest}; expected {config.final_model_sha256}"
        )
    print(f"final CDU model: {final_weight} sha256={final_digest}")
    data = DatasetStore(config.datasets)
    for split, images in data.verify().items():
        counts = data.object_counts(images)
        print(
            f"{split}: images={len(images)}, objects={sum(counts.values())}, "
            f"crack={counts[0]}, patch={counts[1]}, pothole={counts[2]}"
        )
    print(f"acquisition budget: {config.rounds} x {config.images_per_round} = "
          f"{config.rounds * config.images_per_round} acquisitions; "
          f"final train={config.final_train_size}")
    print(
        "sensitivity: full IoU grid="
        f"{list(config.sensitivity_iou)}; paper subset={list(config.sensitivity_report_iou)}; "
        f"confidence={config.sensitivity_conf[0]:.2f}..{config.sensitivity_conf[-1]:.2f} "
        f"({len(config.sensitivity_conf)} points)"
    )


def _freeze_execution_manifest(config: Config) -> Path:
    """Prevent a resumed run from mixing different parameters, data, or code."""
    destination = config.outputs / "execution_manifest.json"
    from active_learning_benchmark.active_learning import prepare_ultralytics
    prepare_ultralytics(config.outputs)

    weight = Path(config.model_source)
    record = config.execution_record()
    record["runtime"] = _runtime()
    record["runtime"]["pretrained_sha256"] = _sha256(weight)
    record["runtime"]["final_cdu_sha256"] = _sha256(Path(config.final_model_source))
    record["dataset_manifests"] = {
        name: _sha256(config.datasets / f"{name}.txt")
        for name in ("train", "valid", "test", "pool")
    }
    sources = [Path(__file__), *(config.root / "active_learning_benchmark").rglob("*.py")]
    record["code_sha256"] = {
        str(path.relative_to(config.root)): _sha256(path)
        for path in sorted(sources)
    }
    content = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(
            f"The effective configuration differs from {destination}. "
            "Use a new run_model directory or restore the recorded configuration."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(content)
    temporary.replace(destination)
    return destination


def _check_accelerator(config: Config) -> None:
    """Refuse an accidental multi-day CPU benchmark unless CPU was explicit."""
    if str(config.device).lower() == "cpu":
        print("accelerator: CPU explicitly requested")
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Do not start this benchmark on CPU by accident. "
            "Verify `nvidia-smi` after reboot, or pass --device cpu intentionally."
        )
    print(f"accelerator: {torch.cuda.get_device_name(0)}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned benchmark; no stage means the complete default pipeline."
    )
    parser.add_argument(
        "stage", nargs="?", default="all",
        choices=("check", "train", "plots", "sensitivity", "all")
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--stop-after-round", type=int, default=None,
        help=(
            "For controlled verification, stop each method after this round. "
            "A later run resumes the existing artifacts."
        ),
    )
    args = parser.parse_args(argv)
    config = _configuration(args)
    atexit.register(_cleanup_amp_probe, config.root)

    if args.stage == "check":
        _check(config)
        _check_runtime(strict=False)
        return

    _check(config)
    print(f"outputs: {config.outputs}")

    if args.stage in ("train", "sensitivity", "all"):
        _check_runtime(strict=True)
        _check_accelerator(config)

    if args.stage in ("train", "all"):
        print(f"execution manifest: {_freeze_execution_manifest(config)}")
        learner = ActiveLearning(config)
        for method in args.methods:
            from active_learning_benchmark.active_learning import release_memory
            release_memory()
            learner.run_method(
                method,
                on_round=update,
                stop_after_round=args.stop_after_round,
            )
            execution_comparison(config.outputs)
            release_memory()

    if args.stage in ("sensitivity", "all"):
        from active_learning_benchmark.sensitivity import run_sensitivity
        run_sensitivity(config, FINAL_CDU_METHOD)
        if args.stage == "sensitivity":
            print(f"sensitivity figures: {sensitivity_figures(config.outputs)['figures']}")

    if args.stage in ("plots", "all"):
        artifacts = paper_figures(config.outputs)
        print(f"paper figures: {artifacts['figures']}")
        if "cluster_visualization" in artifacts:
            print(f"clustering figures: {config.root / 'figures' / 'clustering'}")

if __name__ == "__main__":
    main()
