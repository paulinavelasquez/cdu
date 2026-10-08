"""Random acquisition used as the uncertainty-free reference."""

from .base import Strategy, validate_budget


class Random(Strategy):
    """Select 20 pool images with a reproducible seed for each round."""

    name = "random"

    def select(self, predictions, budget, rng):
        validate_budget(predictions, budget)

        # Apply random.shuffle. Sorting before shuffling makes the seed produce
        # the same batch even if the file system returns a different order.
        available_images = sorted(predictions)
        rng.shuffle(available_images)
        selected_images = available_images[:budget]

        return selected_images
