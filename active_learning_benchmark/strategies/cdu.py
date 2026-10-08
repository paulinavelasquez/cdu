"""CDU and the nine ablations presented in the paper.

The workflow is deliberately separated into the algorithm's five stages:

1. ``build_uncertainty_space`` preserves class-wise uncertainty;
2. ``optimize_n_clusters`` selects the GMM partition by DB index;
3. ``allocate_candidate_quotas`` distributes the candidate set;
4. ``refine_cluster_candidates`` combines uncertainty and diversity;
5. ``retain_final_batch`` retains the most uncertain final batch.

Classes at the end of this file change only the components defined by the
ablation study. This avoids overlapping versions of the same function and
makes the difference between variants explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
import random
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture

from .base import Detection, uncertainty_profile


def build_uncertainty_space(
    predictions: Mapping[str, Sequence[Detection]], classes: int
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Build the ``u_i`` profile and aggregate uncertainty ``U(i)``.

    Each row of ``profiles`` contains the mean uncertainty for ``crack``,
    ``patch``, and ``pothole``. A class with no detections receives zero.
    ``uncertainty`` is the sum of these components and equals the image's DUA score.
    """
    image_ids = tuple(sorted(predictions))

    # ``profiles`` corresponds to ``conj`` in the algorithm: one row per image
    # and one column per class. Keeping components separate preserves class
    # structure before clustering instead of collapsing it into a scalar sum.
    profile_rows: list[tuple[float, ...]] = []
    for image_id in image_ids:
        detections = predictions[image_id]
        class_profile = uncertainty_profile(detections, classes)
        profile_rows.append(class_profile)

    profiles = np.asarray(profile_rows, dtype=float)

    # U(i) remains the DUA sum used to rank candidates and decide the final
    # batch. The GMM receives the complete vector, not this sum.
    image_uncertainty = profiles.sum(axis=1)

    return image_ids, profiles, image_uncertainty


def optimize_n_clusters(
    profiles: np.ndarray,
    classes: int,
    fixed_clusters: int | None = None,
) -> tuple[np.ndarray, int, float | None]:
    """Cluster profiles with GMM and select ``G`` by the lowest DB index.

    The final configuration evaluates ``K`` through ``5K`` components. The
    first ablation fixes ``G=12``. If the remaining pool cannot support two
    valid partitions, all images form one group without changing the budget.
    """
    sample_count = len(profiles)
    distinct_count = len(np.unique(profiles, axis=0))
    if sample_count < 3 or distinct_count < 2:
        return np.zeros(sample_count, dtype=int), 1, None

    counts = (
        [fixed_clusters]
        if fixed_clusters is not None
        else range(classes, 5 * classes + 1)
    )
    best_db_index = float("inf")
    best_cluster_count: int | None = None
    best_labels: np.ndarray | None = None

    for cluster_count in counts:
        largest_valid_count = min(sample_count - 1, distinct_count)
        if cluster_count > largest_valid_count:
            continue

        try:
            clustering = GaussianMixture(
                n_components=cluster_count,
                random_state=0,
                reg_covar=1e-6,
            )
            labels = clustering.fit_predict(profiles)
            actual_clusters = len(set(labels))
            if actual_clusters < 2 or actual_clusters >= sample_count:
                continue
            db_index = float(davies_bouldin_score(profiles, labels))
        except (ValueError, np.linalg.LinAlgError):
            continue

        # A lower DB index represents groups that are more compact internally
        # and farther apart. Ties retain the smaller group count because the
        # range is traversed in ascending order.
        if np.isfinite(db_index) and db_index < best_db_index:
            best_db_index = db_index
            best_cluster_count = cluster_count
            best_labels = labels

    if best_labels is None or best_cluster_count is None:
        return np.zeros(sample_count, dtype=int), 1, None

    return best_labels, best_cluster_count, best_db_index


def allocate_candidate_quotas(
    sizes: Sequence[int], budget: int, rule: str = "sainte_lague"
) -> list[int]:
    """Allocate ``S`` candidates without exceeding group capacity.

    ``sainte_lague`` implements the appendix: proportional floor, an adjusted
    value of 0.67 for nonempty groups with no quota, and odd divisors.
    ``largest`` and ``smallest`` implement the two manual ablation rules by
    visiting groups from largest to smallest or smallest to largest.
    """
    cluster_sizes = list(sizes)
    if (
        not cluster_sizes
        or any(size < 0 for size in cluster_sizes)
        or budget < 0
        or budget > sum(cluster_sizes)
    ):
        raise ValueError("Invalid group sizes or candidate budget")
    if budget == 0:
        return [0] * len(cluster_sizes)

    total_pool_images = sum(cluster_sizes)
    proportional_shares = [
        cluster_size * budget / total_pool_images
        for cluster_size in cluster_sizes
    ]
    quotas = [floor(share) for share in proportional_shares]

    if rule in {"largest", "smallest"}:
        order = sorted(
            range(len(cluster_sizes)),
            key=lambda index: (
                -cluster_sizes[index]
                if rule == "largest"
                else cluster_sizes[index],
                index,
            ),
        )
        while sum(quotas) < budget:
            progressed = False
            for index in order:
                if quotas[index] >= cluster_sizes[index]:
                    continue
                quotas[index] += 1
                progressed = True
                if sum(quotas) == budget:
                    break
            if not progressed:
                raise RuntimeError("Group capacity cannot satisfy the candidate budget")
        return quotas

    if rule != "sainte_lague":
        raise ValueError(f"Unknown allocation rule: {rule}")

    # Sainte-Laguë allocates only the seats left after proportional flooring.
    # Divisors are the next odd values consistent with each group's quota.
    numerators = [
        0.67 if quota == 0 and size else fraction
        for size, quota, fraction in zip(
            cluster_sizes,
            quotas,
            proportional_shares,
        )
    ]
    divisors = [1 if quota == 0 else 2 * quota + 1 for quota in quotas]
    while sum(quotas) < budget:
        available = [
            index
            for index, cluster_size in enumerate(cluster_sizes)
            if quotas[index] < cluster_size
        ]
        winner = max(
            available,
            key=lambda index: (numerators[index] / divisors[index], -index),
        )
        quotas[winner] += 1
        divisors[winner] += 2
    return quotas


def refine_cluster_candidates(
    group: Sequence[int],
    quota: int,
    profiles: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float | None,
) -> list[int]:
    """Select ``s_g`` candidates from a group by uncertainty and diversity.

    Without ``alpha``, the variant uses only ``U(i)``. With ``alpha``, the
    first image is the most uncertain and subsequent images greedily maximize
    the weighted sum of ``U(i)`` and minimum distance to selected profiles.
    """
    if not 0 <= quota <= len(group):
        raise ValueError("The group quota exceeds its capacity")
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("alpha must be in the [0, 1] interval")

    remaining = set(group)
    chosen: list[int] = []
    while len(chosen) < quota:
        if alpha is None:
            # Ablations without Optα select only by the largest U(i).
            picked = max(remaining, key=lambda index: (uncertainty[index], -index))
        elif not chosen:
            # Diversity is undefined for an empty C_g. Explicitly choosing the
            # largest U(i), instead of using infinity, makes the first step
            # deterministic before diversity can be computed.
            picked = max(remaining, key=lambda index: (uncertainty[index], -index))
        else:
            modular_scores: dict[int, float] = {}
            for candidate in remaining:
                distances_to_selected = [
                    np.linalg.norm(profiles[candidate] - profiles[selected])
                    for selected in chosen
                ]
                minimum_diversity_distance = min(distances_to_selected)
                modular_score = (
                    alpha * uncertainty[candidate]
                    + (1 - alpha) * minimum_diversity_distance
                )
                modular_scores[candidate] = modular_score

            picked = max(
                modular_scores,
                key=lambda index: (modular_scores[index], -index),
            )
        remaining.remove(picked)
        chosen.append(picked)
    return chosen


def retain_final_batch(
    candidates: Sequence[int],
    uncertainty: np.ndarray,
    image_ids: Sequence[str],
    budget: int,
) -> list[int]:
    """Retain the ``img_per_int`` largest ``U(i)`` scores."""
    selected = sorted(
        candidates,
        key=lambda index: (-uncertainty[index], image_ids[index]),
    )[:budget]
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError("CDU did not produce a complete batch of unique images")
    return selected


@dataclass(frozen=True)
class CDUSelection:
    images: tuple[str, ...]
    candidate_images: tuple[str, ...]
    alpha: float | None
    cluster_count: int
    db_index: float | None
    candidate_count: int
    cluster_sizes: tuple[int, ...]
    quotas: tuple[int, ...]
    clusters: tuple[int, ...]
    pool_ids: tuple[str, ...]


class CDU:
    """Orchestrate the algorithm shared by the nine ablations without hiding stages.

    Each subclass defines only ``factor``, ``fixed_clusters``,
    ``allocation_rule``, and ``alpha``. The ``select`` method remains unique so
    that variants cannot accidentally execute different active-learning cycles.
    """

    name = "cdu"
    factor = 1
    fixed_clusters: int | None = None
    allocation_rule = "sainte_lague"
    alpha: float | None = None
    classes = 3

    def alpha_for_round(
        self, validation_history: Sequence[dict] = ()
    ) -> float | None:
        return self.alpha

    def select(
        self,
        predictions: Mapping[str, Sequence[Detection]],
        budget: int,
        rng: random.Random | None = None,
        validation_history: Sequence[dict] = (),
    ) -> CDUSelection:
        if budget < 1 or budget > len(predictions):
            raise ValueError("The acquisition budget must fit in the remaining pool")

        # Stage 1 — represent each image by its class-wise uncertainties.
        image_ids, profiles, uncertainty = build_uncertainty_space(
            predictions, self.classes
        )

        # Stage 2 — organize profiles with the GMM of lowest DB index.
        cluster_labels, cluster_count, db_index = optimize_n_clusters(
            profiles, self.classes, self.fixed_clusters
        )

        groups: list[list[int]] = []
        for cluster_id in sorted(set(cluster_labels)):
            images_in_cluster = [
                image_index
                for image_index, assigned_cluster in enumerate(cluster_labels)
                if assigned_cluster == cluster_id
            ]
            groups.append(images_in_cluster)

        # Stage 3 — expand and distribute the candidate set across groups.
        candidate_budget = min(self.factor * budget, len(image_ids))
        cluster_sizes = [len(group) for group in groups]
        candidate_quotas = allocate_candidate_quotas(
            cluster_sizes,
            candidate_budget,
            self.allocation_rule,
        )

        # Stage 4 — refine selection within each group.
        alpha = self.alpha_for_round(validation_history)
        candidates: list[int] = []
        for group, group_quota in zip(groups, candidate_quotas):
            selected_from_cluster = refine_cluster_candidates(
                group=group,
                quota=group_quota,
                profiles=profiles,
                uncertainty=uncertainty,
                alpha=alpha,
            )
            candidates.extend(selected_from_cluster)

        # Stage 5 — diversity forms the candidates; U(i) decides the final batch.
        selected = retain_final_batch(
            candidates, uncertainty, image_ids, budget
        )
        return CDUSelection(
            images=tuple(image_ids[index] for index in selected),
            candidate_images=tuple(image_ids[index] for index in candidates),
            alpha=alpha,
            cluster_count=cluster_count,
            db_index=db_index,
            candidate_count=len(candidates),
            cluster_sizes=tuple(cluster_sizes),
            quotas=tuple(candidate_quotas),
            clusters=tuple(int(value) for value in cluster_labels),
            pool_ids=image_ids,
        )


# The nine variants in the ablation table. Only the attributes below change.


class CDUFM4G12(CDU):
    """Model 1: G=12, FM=4, and remaining seats assigned to larger groups.

    This first ablation configuration builds 80 candidates, uses a fixed number
    of groups, and applies only the U(i) ranking. The published definition
    (12 groups and factor 4) is independent of the number of classes.
    """

    name = "cdu_fm4_g12"
    factor = 4
    fixed_clusters = 12
    allocation_rule = "largest"


class CDUCREC(CDU):
    """Model 2: optimized G and ascending allocation of remaining seats.

    Proportional flooring is shared by the variants; remaining seats visit
    groups from smallest to largest. There is no expansion or modular score.
    """

    name = "cdu_crec"
    allocation_rule = "smallest"


class CDUSL(CDU):
    """Model 3: replace the ascending rule with Sainte-Laguë allocation."""

    name = "cdu_sl"


class CDUSLOptPen(CDU):
    """Model 4: Sainte-Laguë and dynamic alpha guided by the weakest class.

    Alpha starts at 0.55. At each new validation, the class with the lowest
    AP50 is identified; alpha increases by 0.05 if it improved relative to the
    previous round and decreases by 0.05 otherwise. It remains in [0.30, 0.80].
    """

    name = "cdu_sl_optpen"

    def alpha_for_round(self, validation_history=()):
        current_alpha = 0.55
        class_names = ("crack", "patch", "pothole")

        for previous_metrics, current_metrics in zip(
            validation_history,
            validation_history[1:],
        ):
            weakest_class = min(
                class_names,
                key=lambda class_name: current_metrics[class_name],
            )
            weakest_class_improved = (
                current_metrics[weakest_class]
                > previous_metrics[weakest_class]
            )

            adjustment = 0.05 if weakest_class_improved else -0.05
            current_alpha = round(current_alpha + adjustment, 2)
            current_alpha = min(0.80, max(0.30, current_alpha))

        return current_alpha


class CDUSLOpt055(CDU):
    """Model 5: Sainte-Laguë and modular refinement with fixed alpha 0.55."""

    name = "cdu_sl_opt055"
    alpha = 0.55


class CDUSLOpt065(CDU):
    """Model 6: Sainte-Laguë and modular refinement with fixed alpha 0.65."""

    name = "cdu_sl_opt065"
    alpha = 0.65


class CDUFM3SL(CDU):
    """Model 7: FM=3 generates 60 candidates with Sainte-Laguë, without Optα."""

    name = "cdu_fm3_sl"
    factor = 3


class CDUFM4SL(CDU):
    """Model 8: FM=4 generates 80 candidates with Sainte-Laguë, without Optα."""

    name = "cdu_fm4_sl"
    factor = 4


class CDUFM4SLOpt055(CDU):
    """Model 9 and final CDU: FM=4, Sainte-Laguë, and Optα=0.55.

    The modular score diversifies 80 within-group candidates; the 20 largest
    U(i) uncertainties then form the batch sent for annotation.
    """

    name = "cdu_fm4_sl_opt055"
    factor = 4
    alpha = 0.55


PAPER_VARIANTS = (
    CDUFM4G12,
    CDUCREC,
    CDUSL,
    CDUSLOptPen,
    CDUSLOpt055,
    CDUSLOpt065,
    CDUFM3SL,
    CDUFM4SL,
    CDUFM4SLOpt055,
)
VARIANTS = {variant.name: variant for variant in PAPER_VARIANTS}


def make_cdu(name: str) -> CDU:
    try:
        return VARIANTS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown CDU variant: {name}") from error
