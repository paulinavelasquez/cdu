"""Object-oriented training, evaluation, and acquisition cycle.

The rules in ``strategies/`` decide which images enter each batch. This class
runs the shared cycle: resume the previous round, score the pool, record the
selection, update the TXT manifest, train YOLOv9e, evaluate validation/test,
and persist the metrics.
"""

from __future__ import annotations

import csv
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import random

from .config import Config, METHODS
from .data import DatasetStore
from .results import refresh_results
from .strategies import Detection, SIMPLE_STRATEGIES, VARIANTS, make_strategy
from .strategies.base import uncertainty_profile


METRIC_FIELDS = (
    "method", "round", "images", "acquired", "objects", "crack_objects",
    "patch_objects", "pothole_objects",
    "precision", "recall", "map50", "map5095", "fitness",
    "crack_precision", "crack_recall", "crack_ap50", "crack_ap",
    "patch_precision", "patch_recall", "patch_ap50", "patch_ap",
    "pothole_precision", "pothole_recall", "pothole_ap50", "pothole_ap",
    "test_precision", "test_recall", "test_map50", "test_map5095", "test_fitness",
    "test_crack_precision", "test_crack_recall", "test_crack_ap50", "test_crack_ap",
    "test_patch_precision", "test_patch_recall", "test_patch_ap50", "test_patch_ap",
    "test_pothole_precision", "test_pothole_recall", "test_pothole_ap50", "test_pothole_ap",
    "test_results_saved", "alpha", "clusters", "db_index", "candidates",
    "seed", "checkpoint", "checkpoint_sha256",
)
CLASS_NAMES = ("crack", "patch", "pothole")


def release_memory() -> None:
    """Release Python and CUDA memory after each training or evaluation stage."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Remove IPC references accumulated by successive workers.
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError):
                pass
    except ImportError:
        pass
    # Return large blocks released by data loaders and GMM to glibc.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def release_model(model):
    """Remove model references and release RAM and VRAM caches."""
    if model is not None:
        # These components retain the latest tensors and data loaders.
        for attribute in ("predictor", "trainer", "validator", "ckpt"):
            try:
                setattr(model, attribute, None)
            except (AttributeError, RuntimeError):
                pass
        try:
            model.model = None
        except (AttributeError, RuntimeError):
            pass
    del model
    release_memory()
    return None


def prepare_ultralytics(output_root: Path) -> Path:
    """Keep Ultralytics settings and cache inside the run directory."""
    output_root = Path(output_root).resolve()
    settings_root = output_root / "ultralytics"
    settings_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_root))
    from ultralytics import settings
    settings.update({
        "datasets_dir": str(output_root.parent / "datasets"),
        "weights_dir": str(output_root / "pretrained"),
        "runs_dir": str(output_root / "models"),
        "sync": False,
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "raytune": False,
        "wandb": False,
    })
    return settings_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure(metric, prefix: str = "") -> dict:
    """Extract the global and per-class metrics recorded during evaluation."""
    values = {
        f"{prefix}precision": float(metric.box.mp),
        f"{prefix}recall": float(metric.box.mr),
        f"{prefix}map50": float(metric.box.map50),
        f"{prefix}map5095": float(metric.box.map),
        f"{prefix}fitness": float(metric.fitness),
    }
    class_values = {name: (0.0, 0.0, 0.0, 0.0) for name in CLASS_NAMES}
    for position, class_index in enumerate(metric.box.ap_class_index):
        class_values[CLASS_NAMES[int(class_index)]] = (
            float(metric.box.p[position]),
            float(metric.box.r[position]),
            float(metric.box.ap50[position]),
            float(metric.box.ap[position]),
        )
    for name, (precision, recall, ap50, ap) in class_values.items():
        values.update({
            f"{prefix}{name}_precision": precision,
            f"{prefix}{name}_recall": recall,
            f"{prefix}{name}_ap50": ap50,
            f"{prefix}{name}_ap": ap,
        })
    return values


class ActiveLearning:
    """Run the 60 rounds of each strategy sequentially."""

    def __init__(self, config: Config):
        self.config = config.validated()
        prepare_ultralytics(self.config.outputs)
        self.data = DatasetStore(config.datasets)

    def _device_options(self) -> dict:
        return {"device": self.config.device} if self.config.device not in (None, "") else {}

    @staticmethod
    def _read_rows(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _append_row(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})

    def _evaluate(self, method: str, round_index: int, checkpoint: Path,
                  data_yaml: Path) -> dict:
        from ultralytics import YOLO

        detector = YOLO(str(checkpoint))
        try:
            # The historical CSV stores training/validation metrics first and
            # test metrics in the test_* columns. Validation keeps the default
            # confidence used by Ultralytics training; test uses conf=0.01.
            validation = detector.val(
                data=str(data_yaml), split="val", imgsz=self.config.image_size,
                batch=self.config.train_batch, iou=self.config.test_nms_iou,
                plots=False, verbose=False,
                project=str(self.config.outputs / "evaluation" / method),
                name=f"round_{round_index:02d}_validation", exist_ok=True,
                **self._device_options(),
            )
            test = detector.val(
                data=str(data_yaml), split="test", imgsz=self.config.image_size,
                batch=self.config.train_batch, conf=self.config.test_conf,
                iou=self.config.test_nms_iou, save_json=True, save_hybrid=False,
                plots=False, verbose=False,
                project=str(self.config.outputs / "evaluation" / method),
                name=f"round_{round_index:02d}_test", exist_ok=True,
                **self._device_options(),
            )
            values = _measure(validation)
            values.update(_measure(test, "test_"))
            values["test_results_saved"] = str(test.save_dir)
            del validation, test
            return values
        finally:
            detector = release_model(detector)

    def _train(self, method: str, round_index: int,
               manifest: Path) -> tuple[Path, dict]:
        from ultralytics import YOLO

        run_dir = self.config.outputs / "models" / method / f"round_{round_index:02d}"
        checkpoint = run_dir / "weights" / "best.pt"
        last_checkpoint = run_dir / "weights" / "last.pt"
        results_csv = run_dir / "results.csv"
        data_yaml = self.data.configure_yaml(manifest, run_dir / "data.yaml")

        completed_epochs = 0
        if results_csv.is_file():
            with results_csv.open(newline="") as handle:
                completed_epochs = sum(1 for _ in csv.DictReader(handle))

        if completed_epochs < self.config.epochs and last_checkpoint.is_file():
            print(
                f"[{method}] resuming round {round_index:02d}: "
                f"{completed_epochs}/{self.config.epochs} epochs completed"
            )
            detector = YOLO(str(last_checkpoint))
            try:
                detector.train(resume=True, **self._device_options())
            finally:
                detector = release_model(detector)
        elif completed_epochs < self.config.epochs and checkpoint.exists():
            raise RuntimeError(
                f"Partial round without last.pt: {run_dir} has {completed_epochs}/"
                f"{self.config.epochs} epochs. It cannot be accepted as complete."
            )
        elif completed_epochs == 0:
            Path(self.config.model_source).parent.mkdir(parents=True, exist_ok=True)
            detector = YOLO(self.config.model_source)  # reinitialize every round, as stated in the paper
            try:
                detector.train(
                    data=str(data_yaml), epochs=self.config.epochs,
                    lr0=self.config.lr0, lrf=self.config.lrf,
                    batch=self.config.train_batch, imgsz=self.config.image_size,
                    workers=min(self.config.workers, os.cpu_count() or 1),
                    optimizer=self.config.optimizer, patience=self.config.patience,
                    amp=self.config.amp, cache=self.config.cache,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    warmup_epochs=self.config.warmup_epochs,
                    warmup_momentum=self.config.warmup_momentum,
                    warmup_bias_lr=self.config.warmup_bias_lr,
                    box=self.config.box, cls=self.config.cls, dfl=self.config.dfl,
                    hsv_h=self.config.hsv_h, hsv_s=self.config.hsv_s,
                    hsv_v=self.config.hsv_v, degrees=self.config.degrees,
                    translate=self.config.translate, scale=self.config.scale,
                    shear=self.config.shear, perspective=self.config.perspective,
                    flipud=self.config.flipud, fliplr=self.config.fliplr,
                    mosaic=self.config.mosaic, mixup=self.config.mixup,
                    copy_paste=self.config.copy_paste,
                    close_mosaic=self.config.close_mosaic,
                    cos_lr=self.config.cos_lr, rect=self.config.rect,
                    multi_scale=self.config.multi_scale,
                    pretrained=True, resume=False, single_cls=False, val=True,
                    seed=self.config.seed, deterministic=True,
                    project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                    **self._device_options(),
                )
            finally:
                detector = release_model(detector)
        if results_csv.is_file():
            with results_csv.open(newline="") as handle:
                completed_epochs = sum(1 for _ in csv.DictReader(handle))
        if completed_epochs < self.config.epochs:
            raise RuntimeError(
                f"Incomplete training in {run_dir}: "
                f"{completed_epochs}/{self.config.epochs} epochs"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training did not produce {checkpoint}")
        # Ultralytics writes best.pt and last.pt. The benchmark always evaluates
        # best.pt, so last.pt is redundant and would nearly double disk usage.
        if last_checkpoint.is_file() and last_checkpoint != checkpoint:
            last_checkpoint.unlink()
        return checkpoint, self._evaluate(method, round_index, checkpoint, data_yaml)

    def _predict_pool(self, checkpoint: Path, pool: list[Path]) -> dict[str, list[Detection]]:
        from ultralytics import YOLO

        detector = YOLO(str(checkpoint))
        predictions: dict[str, list[Detection]] = {}
        try:
            for start in range(0, len(pool), self.config.acquisition_batch):
                chunk = pool[start:start + self.config.acquisition_batch]
                results = detector.predict(
                    source=[str(image) for image in chunk], batch=len(chunk),
                    imgsz=self.config.image_size, conf=self.config.acquisition_conf,
                    iou=self.config.acquisition_nms_iou, verbose=False,
                    **self._device_options(),
                )
                for image, result in zip(chunk, results):
                    predictions[image.stem] = [
                        Detection(int(cls), float(confidence))
                        for cls, confidence in zip(
                            result.boxes.cls.tolist(), result.boxes.conf.tolist()
                        )
                    ]
                del results
        finally:
            detector = release_model(detector)
        if len(predictions) != len(pool):
            raise RuntimeError("Pool inference did not return one result per image")
        return predictions

    def _select(self, method: str, checkpoint: Path, remaining: list[Path],
                history: list[dict], round_index: int, selection_dir: Path) -> dict:
        # Random does not require inference. Every other strategy receives the
        # same ``image -> detections`` structure.
        if method == "random":
            predictions = {image.stem: [] for image in remaining}
        else:
            predictions = self._predict_pool(checkpoint, remaining)

        strategy = make_strategy(method)
        rng = random.Random(self.config.seed + round_index)

        # Random, Sum, Avg, and DUA return image identifiers directly. CDU also
        # returns G, DB, alpha, quotas, and groups so each algorithm stage can
        # be audited.
        if method in SIMPLE_STRATEGIES:
            selected = strategy.select(predictions, self.config.images_per_round, rng)
            details: dict = {}
        else:
            report = strategy.select(
                predictions, self.config.images_per_round, rng,
                validation_history=history,
            )
            selected = list(report.images)
            details = {
                "alpha": report.alpha,
                "clusters": report.cluster_count,
                "db_index": report.db_index,
                "candidates": report.candidate_count,
                "candidate_images": report.candidate_images,
                "cluster_sizes": report.cluster_sizes,
                "quotas": report.quotas,
            }
            cluster_file = selection_dir / f"clusters_round_{round_index:02d}.csv"
            with cluster_file.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow((
                    "image", "cluster", "crack_uncertainty", "patch_uncertainty",
                    "pothole_uncertainty", "candidate", "selected",
                ))
                chosen = set(selected)
                candidate_images = set(report.candidate_images)
                for image_id, cluster in zip(report.pool_ids, report.clusters):
                    class_profile = uncertainty_profile(predictions[image_id], 3)
                    writer.writerow(
                        (
                            image_id,
                            cluster,
                            *class_profile,
                            int(image_id in candidate_images),
                            int(image_id in chosen),
                        )
                    )
        if len(selected) != self.config.images_per_round or len(set(selected)) != len(selected):
            raise RuntimeError(f"{method} did not select exactly 20 unique images")
        available = {image.stem for image in remaining}
        if not set(selected) <= available:
            raise RuntimeError(f"{method} selected an image outside the remaining pool")
        return {
            "method": method,
            "round": round_index,
            "seed": self.config.seed + round_index,
            "source_checkpoint": str(checkpoint.resolve()),
            "images": selected,
            **details,
        }

    def _metric_row(self, method: str, round_index: int, images: list[Path],
                    checkpoint: Path, details: dict, metrics: dict) -> dict:
        """Build a complete round record before consolidation.

        Object counts come from the cumulative manifest; global and per-class
        metrics come from validation and test evaluation. The consolidated CSV
        expands this record to one row per class.
        """

        counts = self.data.object_counts(images)
        return {
            "method": method,
            "round": round_index,
            "images": len(images),
            "acquired": round_index * self.config.images_per_round,
            "objects": sum(counts.values()),
            "crack_objects": counts[0],
            "patch_objects": counts[1],
            "pothole_objects": counts[2],
            "alpha": details.get("alpha", ""),
            "clusters": details.get("clusters", ""),
            "db_index": details.get("db_index", ""),
            "candidates": details.get("candidates", ""),
            "seed": self.config.seed + round_index,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            **metrics,
        }

    def _baseline(self) -> dict:
        path = self.config.outputs / "active_learning" / "baseline" / "metrics.csv"
        rows = self._read_rows(path)
        if rows:
            missing = set(METRIC_FIELDS) - set(rows[0])
            if missing:
                raise RuntimeError(
                    "The existing run uses an older metrics schema and cannot be resumed "
                    f"safely. Archive run_model and start a clean run. Missing: {sorted(missing)}"
                )
            checkpoint = Path(rows[0]["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            return rows[0]
        images = self.data.read("train")
        checkpoint, metrics = self._train("baseline", 0, self.data.manifest("train"))
        row = self._metric_row("baseline", 0, images, checkpoint, {}, metrics)
        self._append_row(path, row)
        refresh_results(self.config.outputs)
        return row

    def run_method(self, method: str, on_round=None, stop_after_round: int | None = None) -> Path:
        """Run or resume a strategy through the requested round.

        Round ``r`` uses the checkpoint from ``r-1`` to score the remaining
        pool. After recording the 20 selections, training for round ``r``
        restarts from ``weights/yolov9e.pt`` on the cumulative manifest. This
        keeps acquisition separate from detector initialization.
        """

        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        splits = self.data.verify()
        baseline = self._baseline()
        output = self.config.outputs / "active_learning" / method
        selection_dir = output / "selections"
        selection_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output / "metrics.csv"
        if not self._read_rows(metrics_path):
            self._append_row(metrics_path, {**baseline, "method": method})

        target_round = self.config.rounds if stop_after_round is None else stop_after_round
        if not 1 <= target_round <= self.config.rounds:
            raise ValueError(
                f"stop_after_round must be between 1 and {self.config.rounds}; "
                f"received {target_round}"
            )

        for round_index in range(1, target_round + 1):
            # 1. Read only complete, contiguous records. A round with JSON but
            # no metrics can resume without selecting another batch.
            rows = self._read_rows(metrics_path)
            if len(rows) < round_index:
                raise RuntimeError(f"Missing contiguous metrics before round {round_index}")
            selection_file = selection_dir / f"round_{round_index:02d}.json"
            if selection_file.exists():
                choice = json.loads(selection_file.read_text())
            else:
                # 2. Reconstruct previously acquired images from the JSON files.
                used = {
                    image_id
                    for previous_round in range(1, round_index)
                    for image_id in json.loads(
                        (selection_dir / f"round_{previous_round:02d}.json").read_text()
                    )["images"]
                }
                remaining = [image for image in splits["pool"] if image.stem not in used]

                # 3. The OptPen schedule reads only validation AP50 from prior
                # rounds; no test metric guides acquisition.
                history = [
                    {name: float(row[f"{name}_ap50"])
                     for name in ("crack", "patch", "pothole")}
                    for row in rows if row.get("crack_ap50") not in (None, "")
                ]
                choice = self._select(
                    method, Path(rows[round_index - 1]["checkpoint"]), remaining,
                    history, round_index, selection_dir,
                )
                temporary = selection_file.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(choice, indent=2) + "\n")
                temporary.replace(selection_file)

            # 4. Accumulate all selections through the current round. The
            # train_<method>.txt manifest references local data without copies.
            acquired = [
                image_id
                for previous_round in range(1, round_index + 1)
                for image_id in json.loads(
                    (selection_dir / f"round_{previous_round:02d}.json").read_text()
                )["images"]
            ]
            manifest, train_images = self.data.write_train(method, acquired, round_index)
            if len(rows) > round_index:
                continue

            # 5. Train, evaluate, and record the round. ``_train`` always starts
            # from the initial YOLOv9e weights unless it is resuming the same
            # partially completed round.
            checkpoint, metrics = self._train(method, round_index, manifest)
            row = self._metric_row(method, round_index, train_images, checkpoint, choice, metrics)
            self._append_row(metrics_path, row)
            refresh_results(self.config.outputs)
            print(
                f"[{method}] round {round_index:02d}/{self.config.rounds}: "
                f"images={len(train_images)}, objects={row['objects']}, "
                f"test_mAP50={row['test_map50']:.6f}"
            )
            if on_round is not None:
                # 6. Immediately update the progress figures.
                cluster_file = selection_dir / f"clusters_round_{round_index:02d}.csv"
                on_round(method, metrics_path, cluster_file if cluster_file.exists() else None)
            release_memory()

        if target_round < self.config.rounds:
            print(
                f"[{method}] controlled verification stop after round "
                f"{target_round:02d}; the next run will resume from this point."
            )
            return metrics_path

        final_manifest = self.data.read(f"train_{method}")
        if len(final_manifest) != self.config.final_train_size:
            raise RuntimeError(
                f"{method} ended with {len(final_manifest)} images; expected {self.config.final_train_size}"
            )
        # Keep summaries and figures synchronized when a resumed method already
        # has all rounds completed.
        refresh_results(self.config.outputs)
        if on_round is not None:
            cluster_file = selection_dir / f"clusters_round_{self.config.rounds:02d}.csv"
            on_round(method, metrics_path, cluster_file if cluster_file.exists() else None)
        return metrics_path
