"""Diverse Uncertainty Aggregation (DUA), a benchmark comparator."""

from .base import Strategy, uncertainty_profile, validate_budget


class DUA(Strategy):
    """Sum mean uncertainties computed separately for each class.

    Unlike Avg, the means for ``crack``, ``patch``, and ``pothole`` are
    computed first. Summing these components preserves the contribution of
    each present class before producing a scalar score.
    """

    name = "dua"

    def __init__(self, classes=3):
        self.classes = classes

    def select(self, predictions, budget, rng):
        validate_budget(predictions, budget)

        uncertainty_by_image: dict[str, float] = {}
        for image_id, detections in predictions.items():
            # For each class, sum (1 - confidence) and divide by the number of
            # detections. Under the DUA definition, absent classes contribute
            # zero.
            class_uncertainties = uncertainty_profile(
                detections,
                classes=self.classes,
            )
            image_uncertainty = sum(class_uncertainties)
            uncertainty_by_image[image_id] = image_uncertainty

        # Rank by ascending uncertainty and use the image identifier as a
        # deterministic tie-breaker.
        ranked_images = sorted(
            uncertainty_by_image,
            key=lambda image_id: (uncertainty_by_image[image_id], image_id),
        )
        selected_images = ranked_images[-budget:]

        return selected_images
