from .base import Detection, Strategy
from .random import Random
from .sum import Sum
from .avg import Avg
from .dua import DUA
from .cdu import PAPER_VARIANTS, VARIANTS, make_cdu

SIMPLE_STRATEGIES = {"random": Random, "sum": Sum, "avg": Avg, "dua": DUA}


def make_strategy(name, **kwargs):
    if name in SIMPLE_STRATEGIES:
        return SIMPLE_STRATEGIES[name](**kwargs)
    if kwargs:
        raise TypeError("CDU variants are fully specified by their paper-aligned class")
    return make_cdu(name)


__all__ = [
    "Detection", "Strategy", "Random", "Sum", "Avg", "DUA",
    "SIMPLE_STRATEGIES", "PAPER_VARIANTS", "VARIANTS",
    "make_strategy",
]
