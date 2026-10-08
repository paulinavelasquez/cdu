"""TXT manifests and local storage used by the active-learning experiments."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path


EXPECTED_SPLITS = {"train": 252, "valid": 200, "test": 218, "pool": 3607}
SPLIT_IDENTIFIERS = {
    "train": "train_",
    "valid": "valid_",
    "test": "test_",
    "pool": "infer_",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class DatasetStore:
    """Read prepared data splits and update their manifests.

    ``train``, ``valid``, ``test``, and ``pool`` are benchmark inputs. Source
    sequences are separated before this stage and only verified here. During
    acquisition, strategies receive detector predictions only. A pool image's
    label joins training only after that image has been selected.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.cdu_manifests = self.root / "data_CDU"
        self.yaml = self.root / "data.yaml"

    def manifest(self, name: str) -> Path:
        if name.startswith("train_cdu_"):
            return self.cdu_manifests / f"{name}.txt"
        return self.root / f"{name}.txt"

    def _resolve_entry(self, entry: str) -> Path:
        path = Path(entry)
        if path.is_absolute():
            resolved = path.resolve()
        elif path.parent == Path("."):
            # Also accept manifests containing only the file name.
            resolved = (self.images / path.name).resolve()
        else:
            resolved = (self.root / path).resolve()
        if resolved.parent != self.images or not resolved.is_file():
            raise ValueError(
                f"Entry is missing or outside local image storage: {entry}"
            )
        return resolved

    def read(self, name: str) -> list[Path]:
        path = self.manifest(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        images = [self._resolve_entry(line.strip()) for line in path.read_text().splitlines()
                  if line.strip()]
        stems = [image.stem for image in images]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Duplicate image identity in {path}")
        return images

    def write(self, name: str, images: list[Path]) -> Path:
        resolved = [self._resolve_entry(str(image)) for image in images]
        if len(resolved) != len({image.stem for image in resolved}):
            raise ValueError(f"Duplicate image identity in {name}")
        target = self.manifest(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".txt.tmp")
        temporary.write_text("".join(f"images/{image.name}\n" for image in resolved))
        temporary.replace(target)
        return target

    def require_labels(self, images: list[Path]) -> None:
        missing = [image.stem for image in images
                   if not (self.labels / f"{image.stem}.txt").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing YOLO labels for {len(missing)} images: {missing[:5]}"
            )

    def validate_labels(self, images: list[Path]) -> None:
        """Reject invalid YOLO labels before the benchmark starts."""
        for image in images:
            label = self.labels / f"{image.stem}.txt"
            for line_number, line in enumerate(label.read_text().splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"{label}:{line_number}: expected 5 YOLO fields")
                try:
                    cls = int(parts[0])
                    x, y, width, height = map(float, parts[1:])
                except ValueError as error:
                    raise ValueError(f"{label}:{line_number}: nonnumeric label") from error
                if cls not in (0, 1, 2):
                    raise ValueError(f"{label}:{line_number}: invalid class {cls}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and
                        0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(
                        f"{label}:{line_number}: coordinates outside YOLO bounds"
                    )

    def verify(self) -> dict[str, list[Path]]:
        # The four manifests describe a prepared split with source identifiers.
        # This method does not redistribute images; it verifies exclusivity and
        # preserves source sequences across subsets.
        splits = {name: self.read(name) for name in EXPECTED_SPLITS}
        for name, expected in EXPECTED_SPLITS.items():
            if len(splits[name]) != expected:
                raise ValueError(f"{name}.txt: expected {expected}, found {len(splits[name])}")
        owners: dict[str, str] = {}
        for split, images in splits.items():
            for image in images:
                expected_identifier = SPLIT_IDENTIFIERS[split]
                if not image.stem.startswith(expected_identifier):
                    raise ValueError(
                        f"Incompatible identifier: {image.name} does not belong to {split}"
                    )
                if image.stem in owners:
                    raise ValueError(
                        f"Image {image.stem} appears in {owners[image.stem]} and {split}"
                    )
                owners[image.stem] = split
        # Frames from one source sequence must remain in one split so that
        # near-identical scenes do not leak between training and testing.
        sequence_owners: dict[str, str] = {}
        for split, images in splits.items():
            for image in images:
                # The prefix identifies the split but is not part of the video
                # ID. Removing it detects the same segment even if it was
                # mistakenly named for two subsets.
                source_identity = image.stem.removeprefix(SPLIT_IDENTIFIERS[split])
                if "_quadro_" not in source_identity:
                    raise ValueError(
                        f"Could not derive the source sequence from {image.name}"
                    )
                sequence_id = source_identity.split("_quadro_", 1)[0]
                previous = sequence_owners.setdefault(sequence_id, split)
                if previous != split:
                    raise ValueError(
                        f"Sequence {sequence_id} appears in {previous} and {split}"
                    )
        # Confirm required label integrity before starting the full training
        # and acquisition cycle.
        self.require_labels([image for images in splits.values() for image in images])
        self.validate_labels([image for images in splits.values() for image in images])
        return splits

    def normalize_base_manifests(self) -> None:
        """Normalize all four manifests to portable ``images/...`` paths."""
        for name in EXPECTED_SPLITS:
            self.write(name, self.read(name))
        self.configure_yaml(self.manifest("train"))

    def write_train(self, method: str, acquired: list[str], round_index: int) -> tuple[Path, list[Path]]:
        """Create the cumulative round manifest without copying images or labels."""

        initial = self.read("train")
        pool = {image.stem: image for image in self.read("pool")}
        expected = 20 * round_index
        if len(acquired) != expected or len(acquired) != len(set(acquired)):
            raise ValueError(
                f"Round {round_index} requires {expected} distinct acquisitions"
            )
        if not set(acquired) <= set(pool):
            raise ValueError("An acquired image does not belong to pool.txt")
        images = initial + [pool[stem] for stem in acquired]
        if len(images) != 252 + expected:
            raise ValueError("The cumulative training budget is incorrect")
        self.require_labels(images)
        return self.write(f"train_{method}", images), images

    def configure_yaml(self, train_manifest: Path, snapshot: Path | None = None) -> Path:
        """Point YAML to the round manifests and freeze its configuration.

        ``data.yaml`` remains a portable dataset description. During training,
        each round receives a manifest copy and a YAML with absolute paths in
        its output directory, enabling resume without image copies.
        """
        active_manifest = Path(train_manifest).resolve()
        try:
            public_train = active_manifest.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("The training manifest must be inside datasets/") from error

        public_content = (
            "path: '.'\n"
            f"train: '{public_train}'\n"
            "val: 'valid.txt'\n"
            "test: 'test.txt'\n"
            "nc: 3\n"
            "names: ['crack', 'patch', 'pothole']\n"
        )
        self.yaml.write_text(public_content)

        if snapshot is not None:
            snapshot = Path(snapshot)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            frozen_manifest = snapshot.parent / "train.txt"
            round_images = [self._resolve_entry(line.strip())
                            for line in active_manifest.read_text().splitlines() if line.strip()]
            frozen_manifest.write_text("".join(f"{image}\n" for image in round_images))
            active_manifest = frozen_manifest.resolve()
            runtime_content = (
                f"path: '{self.root}'\n"
                f"train: '{active_manifest}'\n"
                f"val: '{self.manifest('valid').resolve()}'\n"
                f"test: '{self.manifest('test').resolve()}'\n"
                "nc: 3\n"
                "names: ['crack', 'patch', 'pothole']\n"
            )
            snapshot.write_text(runtime_content)
            (snapshot.parent / "data.sha256").write_text(
                hashlib.sha256(runtime_content.encode()).hexdigest() + "\n"
            )
            return snapshot
        return self.yaml

    def object_counts(self, images: list[Path]) -> Counter:
        """Count YOLO objects by class in the training manifest."""

        counts: Counter = Counter()
        for image in images:
            label = self.labels / f"{image.stem}.txt"
            for line in label.read_text().splitlines():
                parts = line.split()
                if parts:
                    class_index = int(parts[0])
                    counts[class_index] += 1
        return counts
