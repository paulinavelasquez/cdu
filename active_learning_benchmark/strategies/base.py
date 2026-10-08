"""Structures shared by the acquisition strategies.

Strategies receive only pool predictions. YOLO labels do not enter the
acquisition calculation and are added to the training manifest only after an
image has been selected.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Detection:
    cls: int
    confidence: float


class Strategy:
    """Common interface for Random, Sum, Avg, and DUA."""

    name = "base"

    def select(self, predictions: Mapping[str, Sequence[Detection]], budget: int, rng):
        raise NotImplementedError


def validate_budget(predictions, budget: int) -> None:
    """Confirm that the requested batch fits in the remaining pool."""

    if budget < 1:
        raise ValueError("The acquisition budget must be positive")
    if budget > len(predictions):
        raise ValueError(
            f"The acquisition budget ({budget}) exceeds the remaining pool "
            f"({len(predictions)})"
        )


def uncertainty_profile(
    detections: Sequence[Detection], classes: int = 3
) -> tuple[float, ...]:
    """Compute the mean uncertainty of each class detected in an image.

    For each box ``j``, uncertainty is ``1 - confidence_j``. Boxes are
    accumulated separately by class before their means are computed. An absent
    class receives zero, as defined by the DUA aggregation used in the paper.
    """

    uncertainty_sums = [0.0] * classes
    detections_per_class = [0] * classes

    for detection in detections:
        class_index = detection.cls
        if not 0 <= class_index < classes:
            continue

        object_uncertainty = 1.0 - detection.confidence
        uncertainty_sums[class_index] += object_uncertainty
        detections_per_class[class_index] += 1

    class_uncertainties: list[float] = []
    for class_index in range(classes):
        number_of_detections = detections_per_class[class_index]
        if number_of_detections == 0:
            class_uncertainty = 0.0
        else:
            class_uncertainty = (
                uncertainty_sums[class_index] / number_of_detections
            )
        class_uncertainties.append(class_uncertainty)

    return tuple(class_uncertainties)
