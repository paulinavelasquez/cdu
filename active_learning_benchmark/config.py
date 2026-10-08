"""Fixed configuration for the benchmark described in the paper."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from dataclasses import asdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MAIN_METHODS = ("random", "sum", "avg", "dua")
PAPER_CDU_METHODS = (
    "cdu_fm4_g12",
    "cdu_crec",
    "cdu_sl",
    "cdu_sl_optpen",
    "cdu_sl_opt055",
    "cdu_sl_opt065",
    "cdu_fm3_sl",
    "cdu_fm4_sl",
    "cdu_fm4_sl_opt055",
)
METHODS = MAIN_METHODS + PAPER_CDU_METHODS
DEFAULT_METHODS = METHODS
FINAL_CDU_METHOD = "cdu_fm4_sl_opt055"


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    # Checkpoints used with the stack pinned in requirements.txt (Ultralytics
    # 8.2.28). Hashes prevent silent replacement of the YOLOv9e revision.
    model: str = "weights/yolov9e.pt"
    model_sha256: str = "324e95cb5d403aad6290c167204f0f40ac11b16d79c0b233cf00e1c2b7ff9b5a"
    final_model: str = "weights/cdu_final.pt"
    final_model_sha256: str = "e24264007a93b4f13c9db1a3999baa71b7620a4b968071225e5cf999aedf6955"
    classes: tuple[str, ...] = ("crack", "patch", "pothole")
    rounds: int = 60
    images_per_round: int = 20
    epochs: int = 100
    train_batch: int = 4
    image_size: int = 640
    workers: int = 16
    optimizer: str = "auto"
    patience: int = 100
    amp: bool = True
    cache: bool = False
    lr0: float = 0.001
    lrf: float = 0.001
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    close_mosaic: int = 10
    cos_lr: bool = False
    rect: bool = False
    multi_scale: bool = False
    seed: int = 0
    # Thresholds used to score the acquisition pool.
    acquisition_conf: float = 0.25
    acquisition_nms_iou: float = 0.7
    # The pool is processed one image at a time during acquisition.
    acquisition_batch: int = 1
    # Sensitivity evaluation can batch images because it does not alter acquisition.
    sensitivity_batch: int = 16
    test_conf: float = 0.01
    test_nms_iou: float = 0.7
    device: str | None = None
    # NMS IoU grid used in sensitivity figures and tables.
    sensitivity_iou: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
    sensitivity_report_iou: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
    sensitivity_conf: tuple[float, ...] = tuple(round(i / 100, 2) for i in range(1, 82))
    # Default confusion-matrix criterion in Ultralytics 8.2.28.
    sensitivity_match_iou: float = 0.45
    methods: tuple[str, ...] = field(default=DEFAULT_METHODS)

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def outputs(self) -> Path:
        return self.root / "run_model"

    @property
    def model_source(self) -> str:
        """Resolve the published initial checkpoint deterministically."""
        model = Path(self.model).expanduser()
        if model.is_absolute():
            return str(model)
        if len(model.parts) == 1 and model.suffix == ".pt":
            return str((self.outputs / "pretrained" / model.name).resolve())
        return str((self.root / model).resolve())

    @property
    def final_model_source(self) -> str:
        """Final CDU checkpoint used for sensitivity analysis."""
        model = Path(self.final_model).expanduser()
        return str(model.resolve() if model.is_absolute() else (self.root / model).resolve())

    @property
    def final_train_size(self) -> int:
        return 252 + self.rounds * self.images_per_round

    def validated(self) -> "Config":
        if self.rounds != 60 or self.images_per_round != 20:
            raise ValueError("The paper protocol is exactly 60 rounds x 20 images")
        if self.final_train_size != 1452:
            raise ValueError("The paper protocol must finish with 1,452 training images")
        if not 0 < self.acquisition_conf < 1:
            raise ValueError("acquisition_conf must be in (0, 1)")
        if len(self.model_sha256) != 64:
            raise ValueError("model_sha256 must be a complete SHA-256 digest")
        if len(self.final_model_sha256) != 64:
            raise ValueError("final_model_sha256 must be a complete SHA-256 digest")
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown methods in configuration: {sorted(unknown)}")
        if len(set(self.sensitivity_iou)) != len(self.sensitivity_iou):
            raise ValueError("sensitivity_iou must not contain repeated values")
        historical_iou = (0.5, 0.6, 0.7, 0.8)
        historical_conf = tuple(round(i / 100, 2) for i in range(1, 82))
        if self.sensitivity_iou != historical_iou:
            raise ValueError(f"Historical sensitivity IoU grid is {historical_iou}")
        if self.sensitivity_conf != historical_conf:
            raise ValueError("Historical confidence grid is 0.01..0.81 in 0.01 increments")
        if self.sensitivity_match_iou != 0.45:
            raise ValueError("Ultralytics 8.2.28 confusion-matrix IoU is 0.45")
        historical_report_iou = (0.5, 0.6, 0.7, 0.8)
        if self.sensitivity_report_iou != historical_report_iou:
            raise ValueError(f"Historical filtered IoU table is {historical_report_iou}")
        return self

    def with_overrides(self, **values) -> "Config":
        return replace(self, **{key: value for key, value in values.items() if value is not None}).validated()

    def execution_record(self) -> dict:
        """Portable parameters that must remain fixed during a run."""
        record = asdict(self)
        record.pop("root", None)
        # A subset can be launched later to resume or extend the same benchmark.
        record.pop("methods", None)
        return record

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path or ROOT / "config.json")
        raw = json.loads(path.read_text()) if path.exists() else {}
        aliases = {
            "iterations": "rounds",
            "images_per_iteration": "images_per_round",
            "prediction_chunk": "acquisition_batch",
        }
        for old, new in aliases.items():
            if old in raw and new not in raw:
                raw[new] = raw.pop(old)
        allowed = set(cls.__dataclass_fields__) - {"root"}
        raw = {key: value for key, value in raw.items() if key in allowed}
        for key in ("classes", "methods", "sensitivity_iou", "sensitivity_report_iou", "sensitivity_conf"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(root=path.parent.resolve(), **raw).validated()
