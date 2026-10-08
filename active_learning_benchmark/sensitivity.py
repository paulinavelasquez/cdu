"""Generate the consolidated sensitivity CSV for the final CDU model."""

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .active_learning import prepare_ultralytics, release_memory, release_model
from .data import DatasetStore


CLASSES = ("crack", "patch", "pothole")
RAW_FIELDS = ("image", "class", "confidence", "x1", "y1", "x2", "y2", "width", "height")
SENSITIVITY_FIELDS = (
    "Class", "IoU", "Confidence", "TP", "FP", "FN", "TN",
    "precision", "recall", "mAP50", "mAP50-95", "fitness",
    "cl_p", "cl_r", "cl_ap50", "cl_ap", "Save Path",
)
MATCH_IOU = np.linspace(0.5, 0.95, 10)


def _checkpoint_digest(checkpoint: Path) -> str:
    """Hash a checkpoint without loading the whole YOLO weight file in RAM."""
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def overlap(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def _truth(manifests, raw):
    image_sizes = {row["image"]: (float(row["width"]), float(row["height"])) for row in raw}
    # Empty-prediction images are supplied separately by the caller.
    truth = defaultdict(list)
    for stem, (width, height) in image_sizes.items():
        label = manifests.labels / f"{stem}.txt"
        if not label.exists():
            raise FileNotFoundError(label)
        for line in label.read_text().splitlines():
            values = [float(v) for v in line.split()[:5]]
            if len(values) != 5:
                continue
            cls, cx, cy, w, h = values
            truth[(stem, int(cls))].append((
                (cx - w / 2) * width, (cy - h / 2) * height,
                (cx + w / 2) * width, (cy + h / 2) * height,
            ))
    return truth


def _raw_predictions(config, method, split, nms_iou, checkpoint, manifests):
    digest = _checkpoint_digest(Path(checkpoint))
    out = (config.outputs / "sensitivity" / method /
           f"{split}_iou{nms_iou:.2f}_{digest}_predictions.json")
    if out.exists():
        return json.loads(out.read_text())
    from ultralytics import YOLO

    images = manifests.read(split)
    model = YOLO(str(checkpoint))
    options = {"device": config.device} if config.device else {}
    rows = []
    try:
        for start in range(0, len(images), config.sensitivity_batch):
            chunk = images[start:start + config.sensitivity_batch]
            results = model.predict(
                source=[str(p) for p in chunk], batch=len(chunk), imgsz=config.image_size,
                conf=min(config.sensitivity_conf), iou=nms_iou, verbose=False, **options,
            )
            for image, result in zip(chunk, results):
                height, width = result.orig_shape
                boxes = result.boxes
                if len(boxes) == 0:
                    rows.append(dict(zip(RAW_FIELDS, (image.stem, -1, 0, 0, 0, 0, 0, width, height))))
                else:
                    for cls, confidence, xyxy in zip(
                        boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
                    ):
                        rows.append(dict(zip(RAW_FIELDS, (
                            image.stem, int(cls), float(confidence), *xyxy, width, height,
                        ))))
            del results
    finally:
        model = release_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, separators=(",", ":")))
    return rows


def _match_image(predictions, targets, match_iou):
    """Ultralytics-compatible greedy IoU matching for one image."""
    correct = np.zeros(len(predictions), dtype=bool)
    matches = []
    for target_index, (target_cls, target_box) in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            if target_cls != int(prediction["class"]):
                continue
            score = overlap(
                tuple(float(prediction[key]) for key in ("x1", "y1", "x2", "y2")),
                target_box,
            )
            if score >= match_iou:
                matches.append((score, target_index, prediction_index))
    # Reproduce Ultralytics 8.2.28 DetectionValidator.match_predictions:
    # highest IoU, unique prediction, and then unique annotation.
    if matches:
        ordered = np.asarray(matches, dtype=float)
        ordered = ordered[np.argsort(ordered[:, 0])[::-1]]
        ordered = ordered[np.unique(ordered[:, 2], return_index=True)[1]]
        ordered = ordered[np.unique(ordered[:, 1], return_index=True)[1]]
        correct[ordered[:, 2].astype(int)] = True
    return correct


def _confusion_matrix(predictions_by_image, targets_by_image, match_iou):
    """Reproduce the Ultralytics matrix, including its background row/column."""
    background = len(CLASSES)
    matrix = np.zeros((background + 1, background + 1), dtype=int)
    for image in sorted(set(predictions_by_image) | set(targets_by_image)):
        predictions = predictions_by_image[image]
        targets = targets_by_image[image]
        matches = []
        for target_index, (_, target_box) in enumerate(targets):
            for prediction_index, prediction in enumerate(predictions):
                score = overlap(
                    tuple(float(prediction[key]) for key in ("x1", "y1", "x2", "y2")),
                    target_box,
                )
                if score > match_iou:
                    matches.append((score, target_index, prediction_index))
        if matches:
            ordered = np.asarray(matches, dtype=float)
            ordered = ordered[np.argsort(ordered[:, 0])[::-1]]
            ordered = ordered[np.unique(ordered[:, 2], return_index=True)[1]]
            ordered = ordered[np.argsort(ordered[:, 0])[::-1]]
            ordered = ordered[np.unique(ordered[:, 1], return_index=True)[1]]
        else:
            ordered = np.empty((0, 3), dtype=float)
        used_targets = set(ordered[:, 1].astype(int))
        used_predictions = set(ordered[:, 2].astype(int))
        for _, target_index, prediction_index in ordered:
            target_index = int(target_index)
            prediction_index = int(prediction_index)
            target_class = int(targets[target_index][0])
            predicted_class = int(predictions[prediction_index]["class"])
            matrix[predicted_class, target_class] += 1
        for target_index, (target_class, _) in enumerate(targets):
            if target_index not in used_targets:
                matrix[background, int(target_class)] += 1
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index not in used_predictions:
                matrix[int(prediction["class"]), background] += 1
    return matrix


def _evaluate_all_classes(raw, truth, confidence, match_iou=0.45):
    """Recompute validation metrics from cached inference results."""
    from ultralytics.utils.metrics import ap_per_class

    predictions_by_image = defaultdict(list)
    for row in raw:
        if int(row["class"]) >= 0 and float(row["confidence"]) >= confidence:
            predictions_by_image[row["image"]].append(row)
    targets_by_image = defaultdict(list)
    for (image, cls), boxes in truth.items():
        targets_by_image[image].extend((cls, box) for box in boxes)

    correct_rows, scores, predicted_classes = [], [], []
    for image in sorted(set(predictions_by_image) | set(targets_by_image)):
        predictions = predictions_by_image[image]
        targets = targets_by_image[image]
        correct = np.column_stack([
            _match_image(predictions, targets, threshold) for threshold in MATCH_IOU
        ]) if predictions else np.zeros((0, len(MATCH_IOU)), dtype=bool)
        correct_rows.extend(correct.tolist())
        scores.extend(float(row["confidence"]) for row in predictions)
        predicted_classes.extend(int(row["class"]) for row in predictions)
    target_classes = np.asarray([
        cls for image in sorted(targets_by_image) for cls, _ in targets_by_image[image]
    ], dtype=int)
    predicted_classes = np.asarray(predicted_classes, dtype=int)
    scores = np.asarray(scores, dtype=float)
    true_positive = np.asarray(correct_rows, dtype=bool).reshape(-1, len(MATCH_IOU))

    confusion = _confusion_matrix(predictions_by_image, targets_by_image, match_iou)
    total = int(confusion.sum())
    output = {cls: dict(
        tp=int(confusion[cls, cls]),
        fp=int(confusion[cls, :].sum() - confusion[cls, cls]),
        fn=int(confusion[:, cls].sum() - confusion[cls, cls]),
        tn=0, precision=0.0, recall=0.0, f1=0.0, map50=0.0, map5095=0.0,
    )
              for cls in range(len(CLASSES))}
    for cls in output:
        output[cls]["tn"] = total - output[cls]["tp"] - output[cls]["fp"] - output[cls]["fn"]
    if not len(scores):
        return output
    # The first seven outputs are stable across the historical and current
    # Ultralytics APIs; later versions append plotting curves.
    tp, fp, precision, recall, f1, ap, class_indices = ap_per_class(
        true_positive, scores, predicted_classes, target_classes, plot=False,
        names=dict(enumerate(CLASSES)),
    )[:7]
    for position, cls in enumerate(class_indices):
        cls = int(cls)
        counts = output[cls]
        output[cls] = {
            "tp": counts["tp"], "fp": counts["fp"],
            "fn": counts["fn"], "tn": counts["tn"],
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "map50": float(ap[position, 0]),
            "map5095": float(ap[position].mean()),
        }
    return output


def run_sensitivity(config, method):
    prepare_ultralytics(config.outputs)
    manifests = DatasetStore(config.datasets)
    manifests.verify()
    checkpoint = Path(config.final_model_source)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Official final CDU checkpoint unavailable: {checkpoint}")
    cache = config.outputs / "sensitivity" / method
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for iou in config.sensitivity_iou:
        raw = _raw_predictions(config, method, "valid", iou, checkpoint, manifests)
        truth = _truth(manifests, raw)
        for confidence in config.sensitivity_conf:
            evaluated = _evaluate_all_classes(
                raw, truth, confidence, config.sensitivity_match_iou
            )
            global_precision = float(np.mean([evaluated[cls]["precision"] for cls in range(3)]))
            global_recall = float(np.mean([evaluated[cls]["recall"] for cls in range(3)]))
            global_map50 = float(np.mean([evaluated[cls]["map50"] for cls in range(3)]))
            global_map = float(np.mean([evaluated[cls]["map5095"] for cls in range(3)]))
            fitness = 0.1 * global_map50 + 0.9 * global_map
            save_path = f"run_model/sensitivity/{method}/validation_iou{iou:.2f}"
            for cls, class_name in enumerate(CLASSES):
                metrics = evaluated[cls]
                rows.append({
                    "Class": class_name, "IoU": iou, "Confidence": confidence,
                    # Published convention: FP are unmatched detections; FN are
                    # annotated objects that were not recovered.
                    "TP": metrics["tp"], "FP": metrics["fp"],
                    "FN": metrics["fn"], "TN": metrics["tn"],
                    "precision": global_precision, "recall": global_recall,
                    "mAP50": global_map50, "mAP50-95": global_map,
                    "fitness": fitness, "cl_p": metrics["precision"],
                    "cl_r": metrics["recall"], "cl_ap50": metrics["map50"],
                    "cl_ap": metrics["map5095"], "Save Path": save_path,
                })
        del raw, truth
        release_memory()
    destination = config.outputs / "results" / "sensitivity_results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENSITIVITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination
