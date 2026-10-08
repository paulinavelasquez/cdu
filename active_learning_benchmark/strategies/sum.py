"""Aggregate all detection uncertainties in an image by summation."""

from .base import Strategy, validate_budget


class Sum(Strategy):
    """Prioritize images with the largest sum of ``1 - confidence``.

    Because every box contributes to the sum, a scene with many objects can
    receive a high score even when individual uncertainties are moderate. This
    effect motivates the subsequent comparison with Avg and DUA.
    """

    name = "sum"

    def select(self, predictions, budget, rng):
        validate_budget(predictions, budget)

        uncertainty_by_image: dict[str, float] = {}
        for image_id, detections in predictions.items():
            image_uncertainty = 0.0
            for detection in detections:
                object_uncertainty = 1.0 - detection.confidence
                image_uncertainty += object_uncertainty
            uncertainty_by_image[image_id] = image_uncertainty

        # Sort images by ascending uncertainty and retain the ``budget`` images
        # with the highest scores.
        ranked_images = sorted(
            uncertainty_by_image,
            key=lambda image_id: (uncertainty_by_image[image_id], image_id),
        )
        selected_images = ranked_images[-budget:]

        return selected_images
