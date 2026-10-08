"""Aggregate an image's detection uncertainties by their mean."""

from .base import Strategy, validate_budget


class Avg(Strategy):
    """Prioritize the highest mean uncertainty without separating classes."""

    name = "avg"

    def select(self, predictions, budget, rng):
        validate_budget(predictions, budget)

        uncertainty_by_image: dict[str, float] = {}
        for image_id, detections in predictions.items():
            if not detections:
                uncertainty_by_image[image_id] = 0.0
                continue

            uncertainty_sum = 0.0
            for detection in detections:
                uncertainty_sum += 1.0 - detection.confidence

            uncertainty_by_image[image_id] = uncertainty_sum / len(detections)

        # Rank in ascending order, then retain the highest-scoring images.
        ranked_images = sorted(
            uncertainty_by_image,
            key=lambda image_id: (uncertainty_by_image[image_id], image_id),
        )
        selected_images = ranked_images[-budget:]

        return selected_images
