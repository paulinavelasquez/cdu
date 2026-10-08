"""Figures and tables derived from the results and methodology snapshot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

from .sensitivity_visuals import build_sensitivity_artifacts


CLASSES = ("crack", "patch", "pothole")
CLASS_STYLE = {"crack": ("red", "^"), "patch": ("green", "s"), "pothole": ("black", "o")}
MAIN_METHODS = (
    ("random", "Random", "black", "o"), ("sum", "Sum", "blue", "D"),
    ("avg", "Avg", "orange", "^"), ("DUA", "DUA", "green", "s"),
    ("CDU_FM4_SL_OPT055", "CDU", "red", "x"),
)
PAPER_CDU = (
    ("CDU_FM4G12", "CDU_FM4G12", "black"),
    ("CDU_creciente", "CDU_CREC", "pink"),
    ("CDU_SL", "CDU_SL", "grey"),
    ("CDU_SL_OptPen", "CDU_SL_OptαPen", "green"),
    ("CDU_SL_Opt055", "CDU_SL_Optα0.55", "blue"),
    ("CDU_SL_Opt065", "CDU_SL_Optα0.65", "purple"),
    ("CDU_FM3_SL", "CDU_FM3_SL", "orange"),
    ("CDU_FM4_SL", "CDU_FM4_SL", "brown"),
    ("CDU_FM4_SL_OPT055", "CDU_FM4_SL_Optα0.55", "red"),
)
PAPER_METHODS = (
    "basic", "random", "sum", "avg", "DUA",
    *(method for method, _, _ in PAPER_CDU),
)

# Highest test mAP@50 in the complete experimental record used for the
# published normalization. Figures show only the methods discussed in the
# paper, while AUC retains this scale and integrates all available rounds.
PUBLISHED_AUC_CEILING = 0.4269996822843858


def _rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _paper_frame(ax):
    for name in ("top", "right", "bottom", "left"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color("black")
        ax.spines[name].set_linewidth(0.8)
        ax.spines[name].set_alpha(0.35 if name in ("top", "right") else 1.0)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both", colors="black")
    ax.grid(axis="both", which="major", linewidth=0.5, alpha=0.6)


def _save(fig, destination, *, tight=True):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(destination.with_suffix(".pdf"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _auc(x, y, ceiling):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or ceiling <= 0 or x[-1] == x[0]:
        return float("nan")
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x) / ((x[-1] - x[0]) * ceiling))


def _global_auc_ceiling(data):
    """Return the common empirical ceiling used by the published AUCs."""
    published = data[data.method.isin(PAPER_METHODS) & (data.method != "basic")]
    observed = float(pd.to_numeric(published.test_mAP50, errors="coerce").max())
    if not np.isfinite(observed) or observed <= 0:
        raise ValueError("The AUC trajectories could not be validated")
    if observed > PUBLISHED_AUC_CEILING + 1e-12:
        raise ValueError("The observed mAP@50 exceeds the published normalizer")
    return PUBLISHED_AUC_CEILING


def _validate_active_data(data):
    methods = tuple(data.method.drop_duplicates())
    if methods != PAPER_METHODS:
        raise ValueError(f"Unexpected method order in active_learning_results.csv: {methods}")
    duplicated = data[data.duplicated(["method", "iteration", "class"], keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "active_learning_results.csv must contain one observation per "
            "method, iteration, and class"
        )

    baseline = data[data.method == "basic"]
    if set(baseline["class"]) != set(CLASSES) or len(baseline) != len(CLASSES):
        raise ValueError("The baseline must contain exactly one row per class")

    expected = {
        (method, iteration, class_name)
        for method in methods
        if method != "basic"
        for iteration in range(60)
        for class_name in CLASSES
    }
    measured = {
        (row.method, int(row.iteration), row["class"])
        for _, row in data[data.method != "basic"].iterrows()
    }
    missing = expected - measured
    documented_gap = {
        ("CDU_FM4_SL", 48, class_name) for class_name in CLASSES
    }
    if missing not in (set(), documented_gap):
        sample = sorted(missing)[:6]
        raise ValueError(
            "Unexpected missing iterations/classes in "
            f"active_learning_results.csv: {sample}"
        )
    unexpected = measured - expected
    if unexpected:
        raise ValueError(
            "Iterations/classes outside the protocol in "
            f"active_learning_results.csv: {sorted(unexpected)[:6]}"
        )

    bounded = (
        "precision", "recall", "mAP50", "mAP50-95", "fitness",
        "cl_p", "cl_r", "cl_ap50", "cl_ap", "test_precision",
        "test_recall", "test_mAP50", "test_mAP50-95", "test_fitness",
        "test_cl_p", "test_cl_r", "test_cl_ap50", "test_cl_ap",
    )
    for field in bounded:
        values = pd.to_numeric(data[field], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"Invalid or missing metric in {field}")

    expected_test_objects = {"crack": 60, "patch": 250, "pothole": 77}
    if not data.test_instances.eq(data["class"].map(expected_test_objects)).all():
        raise ValueError("test_instances must represent the fixed test set")

    active = data[data.method != "basic"]
    if not pd.to_numeric(active.infer, errors="coerce").eq(20).all():
        raise ValueError("Each round must acquire exactly 20 images")

    factors = {
        "CDU_FM4G12": 4, "CDU_creciente": 1, "CDU_SL": 1,
        "CDU_SL_OptPen": 1, "CDU_SL_Opt055": 1, "CDU_SL_Opt065": 1,
        "CDU_FM3_SL": 3, "CDU_FM4_SL": 4, "CDU_FM4_SL_OPT055": 4,
    }
    for method, factor in factors.items():
        values = pd.to_numeric(data.loc[data.method == method, "mult_factor"])
        if not values.eq(factor).all():
            raise ValueError(f"Inconsistent multiplication factor in {method}")

    fixed_alpha = {
        "CDU_SL_Opt055": .55,
        "CDU_SL_Opt065": .65,
        "CDU_FM4_SL_OPT055": .55,
    }
    for method, alpha in fixed_alpha.items():
        values = pd.to_numeric(data.loc[data.method == method, "alpha"])
        if not np.isclose(values, alpha).all():
            raise ValueError(f"Inconsistent fixed alpha in {method}")
    dynamic_alpha = pd.to_numeric(
        data.loc[data.method == "CDU_SL_OptPen", "alpha"], errors="coerce"
    )
    if dynamic_alpha.isna().any() or not dynamic_alpha.between(.30, .80).all():
        raise ValueError("Invalid alpha schedule in CDU_SL_OptPen")
    fixed_clusters = pd.to_numeric(
        data.loc[data.method == "CDU_FM4G12", "clusters"]
    )
    if not fixed_clusters.eq(12).all():
        raise ValueError("CDU_FM4G12 must retain G=12")


def _validate_sensitivity_data(sensitivity):
    if len(sensitivity) != 972:
        raise ValueError(f"sensitivity_results.csv must have 972 rows; found {len(sensitivity)}")
    if set(sensitivity.IoU) != {0.5, 0.6, 0.7, 0.8} or sensitivity.Confidence.nunique() != 81:
        raise ValueError("The sensitivity grid must contain 4 IoUs × 81 confidences")
    for field in ("cl_p", "cl_r", "cl_ap50", "cl_ap"):
        if not sensitivity[field].between(0, 1).all():
            raise ValueError(f"Metric outside [0, 1] in {field}")
    expected_objects = {"crack": 58, "patch": 223, "pothole": 25}
    recovered = sensitivity.TP + sensitivity.FN
    expected = sensitivity["Class"].map(expected_objects)
    if expected.isna().any() or not recovered.eq(expected).all():
        raise ValueError(
            "TP + FN must match the annotated total for each validation class"
        )


def method_plots(method, metrics_path):
    """Update a strategy's curves after each completed round."""
    rows = _rows(metrics_path)
    if not rows:
        return
    output = Path(metrics_path).parent / "figures"
    images = np.array([int(row["images"]) for row in rows])
    objects = np.array([int(row["objects"]) for row in rows])
    map50 = np.array([float(row["test_map50"]) for row in rows])
    for x, label, filename in (
        (images, "Number of labeled images", "graph_iterations_vs_map50"),
        (objects, "Number of labeled objects", "graph_instances_vs_map50"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, map50, marker="o", linewidth=2)
        ax.set(xlabel=label, ylabel="mAP@50", title=method)
        _paper_frame(ax)
        _save(fig, output / filename)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls, (color, marker) in CLASS_STYLE.items():
        ax.plot(images, [float(row[f"test_{cls}_ap50"]) for row in rows],
                color=color, marker=marker, label=cls)
    ax.set(xlabel="Number of labeled images", ylabel="mAP@50", title=method)
    ax.legend()
    _paper_frame(ax)
    _save(fig, output / "class_ap50")


def cluster_plot(cluster_csv, output=None):
    """Plot uncertainty profiles by group; axes are classes, not an embedding."""
    cluster_csv = Path(cluster_csv)
    if not cluster_csv.exists() or not (rows := _rows(cluster_csv)):
        return
    output = output or cluster_csv.with_name(cluster_csv.stem + "_scatter")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pairs = (("crack_uncertainty", "patch_uncertainty"),
             ("crack_uncertainty", "pothole_uncertainty"),
             ("patch_uncertainty", "pothole_uncertainty"))
    groups = sorted({int(row["cluster"]) for row in rows})
    for ax, (x, y) in zip(axes, pairs):
        for group in groups:
            values = [row for row in rows if int(row["cluster"]) == group]
            ax.scatter([float(row[x]) for row in values], [float(row[y]) for row in values],
                       s=5, alpha=.35, label=f"G{group}")
        chosen = [row for row in rows if int(row["selected"])]
        ax.scatter([float(row[x]) for row in chosen], [float(row[y]) for row in chosen],
                   s=35, facecolors="none", edgecolors="black", linewidths=.8)
        ax.set(xlabel=x.replace("_uncertainty", ""), ylabel=y.replace("_uncertainty", ""))
        _paper_frame(ax)
    axes[0].legend(fontsize=6, ncol=2)
    _save(fig, output)


def execution_comparison(root):
    """Compare available runs without duplicating the consolidated table."""
    root = Path(root)
    series = [(path.parent.name, _rows(path))
              for path in sorted((root / "active_learning").glob("*/metrics.csv"))
              if path.parent.name != "baseline"]
    series = [(name, rows) for name, rows in series if rows]
    if not series:
        return
    output = root / "figures" / "execution"
    for axis, xlabel, filename in (
        ("images", "Number of labeled images", "graph_iterations_vs_map50"),
        ("objects", "Number of labeled objects", "graph_instances_vs_map50"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, rows in series:
            ax.plot([int(row[axis]) for row in rows],
                    [float(row["test_map50"]) for row in rows], label=name, linewidth=2)
        ax.set(xlabel=xlabel, ylabel="mAP@50")
        ax.legend(ncol=2, fontsize=8)
        _paper_frame(ax)
        _save(fig, output / filename)


def _checkpoint_rows(data, method):
    rows = data[(data.method == method) & data.iteration.notna() &
                (data["class"] == "crack")].sort_values("iteration")
    return rows[((rows.iteration.astype(int) + 1) % 6) == 0]


def _main_curves(data, figures):
    ceiling = _global_auc_ceiling(data)
    baseline_map = float(data.loc[data.method == "basic", "test_mAP50"].iloc[0])
    curves = []
    for method, label, color, marker in MAIN_METHODS:
        all_rows = data[(data.method == method) & data.iteration.notna() &
                        (data["class"] == "crack")].sort_values("iteration")
        curves.append({
            "method_key": method,
            "method": label,
            "color": color,
            "marker": marker,
            "normalized_auc": _auc(
                all_rows.iteration, all_rows.test_mAP50, ceiling
            ),
        })
    curves.sort(key=lambda row: row["normalized_auc"], reverse=True)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for curve in curves:
        method = curve["method_key"]
        label = curve["method"]
        color = curve["color"]
        marker = curve["marker"]
        auc_value = curve["normalized_auc"]
        shown = _checkpoint_rows(data, method)
        x = np.r_[0, (shown.iteration.to_numpy(float) + 1) * 20]
        y = np.r_[baseline_map, shown.test_mAP50.to_numpy(float)]
        ax1.plot(x, y, label=f"{label} (AUC = {auc_value:.4f})", color=color,
                 marker=marker, markersize=8, linewidth=2)
        class_rows = data[(data.method == method) & data.iteration.isin(shown.iteration)]
        objects = class_rows.groupby("iteration").instances.sum().sort_index()
        global_map = shown.set_index("iteration").loc[objects.index, "test_mAP50"]
        ax2.plot(objects.to_numpy(), global_map.to_numpy(), label=label, color=color,
                 marker=marker, markersize=8, linestyle="")
    for ax in (ax1, ax2):
        _paper_frame(ax)
    ax1.set(xlabel="Number of acquired images", ylabel="mAP@50", xlim=(0, 1200),
            ylim=(0, ceiling))
    ax1.set_xticks(np.arange(0, 1201, 120))
    ax1.legend(loc="lower right", fontsize=14, frameon=True)
    ax2.set(xlabel="Number of labeled objects", ylabel="mAP@50")
    ax2.legend(loc="upper left", fontsize=14, frameon=True)
    _save(fig1, figures / "graph_iterations_vs_map50")
    _save(fig2, figures / "graph_instances_vs_map50")
    return pd.DataFrame(curves)[["method", "normalized_auc"]]


def _main_table(data):
    ceiling = _global_auc_ceiling(data)
    definitions = (("basic", "Baseline"),) + tuple((m, label) for m, label, _, _ in MAIN_METHODS)
    records = []
    for method, label in definitions:
        rows = data[data.method == method]
        if method == "basic":
            final, auc_value, images = rows, np.nan, 252
        else:
            rows = rows[rows.iteration.notna()]
            final = rows[rows.iteration == rows.iteration.max()]
            trajectory = rows[rows["class"] == "crack"].sort_values("iteration")
            auc_value = _auc(trajectory.iteration, trajectory.test_mAP50, ceiling)
            images = 1452
        representative = final[final["class"] == "crack"].iloc[-1]
        records.append({
            "method": label, "images": images, "objects": int(final.instances.sum()),
            "precision": representative.test_precision, "mAP50": representative.test_mAP50,
            **{f"mAP50_{cls}": final[final["class"] == cls].iloc[-1].test_cl_ap50 for cls in CLASSES},
            "normalized_auc": auc_value,
        })
    return pd.DataFrame(records)


def _relationship(data, method, destination):
    subset = data[(data.method == method) & data.iteration.notna()].copy()
    subset = subset[((subset.iteration.astype(int) + 1) % 6) == 0]
    subset["labeled_images"] = (subset.iteration.astype(int) + 1) * 20
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()
    iterations = np.sort(subset.labeled_images.unique())
    bar_width = 25
    for offset, cls in enumerate(CLASSES, start=-1):
        color, marker = CLASS_STYLE[cls]
        rows = subset[subset["class"] == cls].sort_values("labeled_images")
        ax1.plot(rows.labeled_images, rows.test_cl_ap50, label=f"mAP {cls}", color=color,
                 marker=marker, markersize=14, markerfacecolor="none", linewidth=1)
        for x, y in zip(rows.labeled_images, rows.test_cl_ap50):
            ax1.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9,
                     bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        ax2.bar(rows.labeled_images + offset * bar_width, rows.instances, bar_width,
                label=f"instances {cls}", color=color, alpha=.5)
    shared_ymax = float(data[
        data.method.isin(("DUA", "CDU_FM4_SL_OPT055"))
    ].test_cl_ap50.max()) + .02
    ax1.set_xticks(iterations)
    ax1.set(
        xlabel="Number of acquired images", ylabel="mAP@50",
        xlim=(0, 1220), ylim=(0, shared_ymax),
    )
    ax2.set_ylabel("Number of labeled objects")
    _paper_frame(ax1)
    handles = ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()
    ax1.legend(handles[0][0] + handles[1][0], handles[0][1] + handles[1][1],
               loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=6)
    _save(fig, destination)


def _auc_crack(data, figures):
    subset = data[(data["class"] == "crack") & data.iteration.notna()].copy()
    methods = PAPER_CDU
    subset = subset[subset.method.isin([name for name, _, _ in methods])]
    baseline_crack = float(data.loc[
        (data.method == "basic") & (data["class"] == "crack"), "test_cl_ap50"
    ].iloc[0])

    # The normalization belongs to the compared set: its ceiling is the
    # highest crack mAP@50 observed across the nine official ablations.
    ceiling = float(subset.test_cl_ap50.max())
    records, lines, labels, plotted = [], [], [], []
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=.12, right=.98, bottom=.08, top=.58)
    for method, label, color in methods:
        rows = subset[subset.method == method].sort_values("iteration").copy()
        value = _auc(rows.iteration, rows.test_cl_ap50, ceiling)
        records.append({"method": method, "normalized_auc_crack": value,
                        "normalizer": ceiling, "available_iterations": len(rows)})
        shown = rows[((rows.iteration.astype(int) + 1) % 6) == 0]
        x = np.r_[0, (shown.iteration.to_numpy(float) + 1) * 20]
        y = np.r_[baseline_crack, shown.test_cl_ap50.to_numpy(float)]
        line, = ax.plot(x, y, color=color, linewidth=2, alpha=1,
                        label=f"{label} (AUC = {value:.4f})")
        lines.append(line); labels.append(line.get_label()); plotted.extend(y)
    ax.set(xlabel="Number of acquired images", ylabel="mAP@50", xlim=(0, 1200))
    ax.set_xticks(np.arange(0, 1201, 120))
    ymax = max(plotted) if plotted else ceiling
    ax.set_ylim(0, ymax + max(.0002, ymax * .03))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    _paper_frame(ax)
    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(.55, .98), ncol=2,
               fontsize=12, frameon=True, columnspacing=1.2, handlelength=1.8,
               handletextpad=.6, labelspacing=.9, borderaxespad=0)
    _save(fig, figures / "AUC_crack", tight=False)
    return pd.DataFrame(records)


def _variant_table(data):
    ceiling = _global_auc_ceiling(data)
    records = []
    for method, label, _ in PAPER_CDU:
        rows = data[(data.method == method) & data.iteration.notna()].copy()
        rows = rows.drop_duplicates(["iteration", "class"], keep="last")
        rows = rows.sort_values("iteration")
        final = rows[rows.iteration == rows.iteration.max()]
        representative = final[final["class"] == "crack"].iloc[-1]
        trajectory = rows[rows["class"] == "crack"]
        records.append({
            "variant": label, "objects": int(final.instances.sum()),
            "precision": representative.test_precision, "mAP50": representative.test_mAP50,
            **{f"mAP50_{cls}": final[final["class"] == cls].iloc[-1].test_cl_ap50 for cls in CLASSES},
            "normalized_auc": _auc(trajectory.iteration, trajectory.test_mAP50, ceiling),
            "available_iterations": int(trajectory.iteration.nunique()),
        })
    return pd.DataFrame(records)


def active_learning_figures(root):
    """Reproduce active-learning figures and DataFrames."""
    root = Path(root).resolve()
    results = root / "results"
    active_csv = results / "active_learning_results.csv"
    if not active_csv.is_file():
        raise FileNotFoundError(f"Required file is missing: {active_csv}")
    data = pd.read_csv(active_csv, low_memory=False)
    _validate_active_data(data)
    figures = root.parent / "figures" / "paper"
    figures.mkdir(parents=True, exist_ok=True)
    active_auc = _main_curves(data, figures)
    main_table = _main_table(data)
    _relationship(data, "DUA", figures / "bar_DUA")
    _relationship(data, "CDU_FM4_SL_OPT055", figures / "bar_CDU")
    variants = _variant_table(data)
    crack_auc = _auc_crack(data, figures)
    return {
        "figures": figures,
        "active_learning_auc": active_auc,
        "tab_results": main_table,
        "tab_results_model": variants,
        "auc_crack": crack_auc,
    }


def sensitivity_figures(root):
    """Reproduce sensitivity curves, heatmaps, and DataFrames."""
    root = Path(root).resolve()
    sensitivity_csv = root / "results" / "sensitivity_results.csv"
    if not sensitivity_csv.is_file():
        raise FileNotFoundError(f"Required file is missing: {sensitivity_csv}")
    sensitivity = pd.read_csv(sensitivity_csv, low_memory=False)
    _validate_sensitivity_data(sensitivity)
    figures = root.parent / "figures" / "paper"
    figures.mkdir(parents=True, exist_ok=True)
    return {"figures": figures, **build_sensitivity_artifacts(sensitivity, figures)}


def paper_figures(root):
    """Regenerate the paper results and the explanatory CDU visualization."""
    root = Path(root).resolve()
    artifacts = {**active_learning_figures(root), **sensitivity_figures(root)}

    # The 3D panels explain the selection mechanism but are not paper figures,
    # so they are stored in a separate directory.
    cluster_output = root.parent / "figures" / "clustering"
    recorded_round = (
        root / "active_learning" / "cdu_fm4_sl_opt055" /
        "selections" / "clusters_round_36.csv"
    )
    if recorded_round.is_file():
        from .cluster_visualization import build_cluster_selection_visualizations
        artifacts["cluster_visualization"] = build_cluster_selection_visualizations(
            recorded_round,
            cluster_output,
        )
    else:
        snapshot = root.parent / "notebooks" / "data" / "uncertainty_profiles.pkl"
        if snapshot.is_file():
            from .cluster_visualization import build_cluster_snapshot_visualizations
            artifacts["cluster_visualization"] = build_cluster_snapshot_visualizations(
                snapshot,
                cluster_output,
            )
    return artifacts


def update(method, metrics_path, cluster_csv=None):
    method_plots(method, metrics_path)
    if cluster_csv:
        cluster_plot(cluster_csv)
        cluster_csv = Path(cluster_csv)
        if method == "cdu_fm4_sl_opt055" and cluster_csv.stem == "clusters_round_36":
            from .cluster_visualization import build_cluster_selection_visualizations
            repository = Path(metrics_path).resolve().parents[3]
            build_cluster_selection_visualizations(
                cluster_csv,
                repository / "figures" / "clustering",
            )
    execution_comparison(Path(metrics_path).parents[2])


__all__ = [
    "active_learning_figures", "sensitivity_figures", "paper_figures",
    "execution_comparison", "method_plots", "cluster_plot", "update",
]
