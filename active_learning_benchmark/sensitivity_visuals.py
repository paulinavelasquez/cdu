"""Analyze and visualize sensitivity to IoU and confidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CLASS_METRICS_CONFIG = (
    {"column": "cl_ap50", "label": "AP50", "goal": "max"},
    {"column": "cl_p", "label": "P", "goal": "max"},
    {"column": "cl_r", "label": "R", "goal": "max"},
    {"column": "cl_f1", "label": "F1", "goal": "max"},
    {"column": "tp", "label": "TP", "goal": "max"},
    {"column": "tpr", "label": "TPR", "goal": "max"},
    {"column": "fn", "label": "FN", "goal": "min"},
)
GENERAL_METRICS_CONFIG = (
    {"column": "map50", "label": "AP50", "goal": "max"},
    {"column": "precision", "label": "P", "goal": "max"},
    {"column": "recall", "label": "R", "goal": "max"},
    {"column": "f1", "label": "F1", "goal": "max"},
    {"column": "tp", "label": "TP", "goal": "max"},
    {"column": "tpr", "label": "TPR", "goal": "max"},
    {"column": "fn", "label": "FN", "goal": "min"},
)
OPT_SCORE_LABELS = ("AP50", "P", "TPR")
WEIGHTS = {label: 1 / 3 for label in OPT_SCORE_LABELS}
MIN_TP_RATIO = 0.35
METRIC_TOLERANCE = 1e-12
REFERENCE_IOU = 0.70

COLOR_MAP = {
    "crack": ("red", "darkred", "lightcoral", "firebrick"),
    "patch": ("green", "darkgreen", "lightgreen", "forestgreen"),
    "pothole": ("black", "gray", "dimgray", "darkgray"),
}
LINESTYLE_MAP = {0.5: "-", 0.6: "--", 0.7: ":", 0.8: "-."}
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures" / "paper"


def prepare_metrics(source: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from the consolidated CSV."""
    columns = {
        "Class": "class", "IoU": "iou", "Confidence": "confidence",
        "TP": "tp", "FP": "fp", "FN": "fn", "TN": "tn",
        "precision": "precision", "recall": "recall", "mAP50": "map50",
        "cl_p": "cl_p", "cl_r": "cl_r", "cl_ap50": "cl_ap50",
    }
    available = [column for column in columns if column in source.columns]
    data = source[available].rename(columns={key: columns[key] for key in available}).copy()
    data["precision_from_counts"] = np.where(
        data.tp + data.fp > 0, data.tp / (data.tp + data.fp), 0.0)
    data["recall_from_counts"] = np.where(
        data.tp + data.fn > 0, data.tp / (data.tp + data.fn), 0.0)
    data["tpr_counts"] = data.recall_from_counts.replace([np.inf, -np.inf], 0).fillna(0)
    data["tpr"] = data.cl_r.replace([np.inf, -np.inf], 0).fillna(0)
    denominator = data.cl_p + data.cl_r
    data["cl_f1"] = np.where(
        denominator > 0, 2 * data.cl_p * data.cl_r / denominator, 0.0)
    data["cl_f1"] = data.cl_f1.replace([np.inf, -np.inf], 0).fillna(0)
    return data


def build_general_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["iou", "confidence"], as_index=False).agg(
        {"tp": "sum", "fp": "sum", "fn": "sum", "map50": "mean"})
    grouped["precision"] = np.where(
        grouped.tp + grouped.fp > 0, grouped.tp / (grouped.tp + grouped.fp), 0.0)
    grouped["recall"] = np.where(
        grouped.tp + grouped.fn > 0, grouped.tp / (grouped.tp + grouped.fn), 0.0)
    grouped["f1"] = np.where(
        grouped.precision + grouped.recall > 0,
        2 * grouped.precision * grouped.recall / (grouped.precision + grouped.recall), 0.0)
    grouped["tpr"] = grouped["recall"]
    grouped["class"] = "general"
    return grouped


def normalize_metric(series: pd.Series, goal: str) -> pd.Series:
    minimum, maximum = float(series.min()), float(series.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.ones(len(series)), index=series.index)
    return ((series - minimum) / (maximum - minimum) if goal == "max"
            else (maximum - series) / (maximum - minimum))


def identify_pareto_front(data: pd.DataFrame, metrics_config) -> np.ndarray:
    transformed = []
    for config in metrics_config:
        values = data[config["column"]].to_numpy(dtype=float)
        transformed.append(values if config["goal"] == "max" else -values)
    values = np.column_stack(transformed)
    is_pareto = np.ones(len(values), dtype=bool)
    for index, candidate in enumerate(values):
        if not is_pareto[index]:
            continue
        dominates = np.all(values >= candidate, axis=1) & np.any(values > candidate, axis=1)
        dominates[index] = False
        if dominates.any():
            is_pareto[index] = False
    return is_pareto


def score_configurations(data: pd.DataFrame, metrics_config) -> pd.DataFrame:
    scored = data.copy()
    score_configs = [config for config in metrics_config if config["label"] in OPT_SCORE_LABELS]
    for config in metrics_config:
        scored[f"{config['column']}_norm"] = normalize_metric(
            scored[config["column"]], config["goal"])
    scored["wins"] = 0
    for config in score_configs:
        metric = config["column"]
        best = scored[metric].max() if config["goal"] == "max" else scored[metric].min()
        scored["wins"] += np.isclose(scored[metric], best).astype(int)
    scored["weighted_score"] = sum(
        WEIGHTS[config["label"]] * scored[f"{config['column']}_norm"]
        for config in score_configs)
    scored["pareto_optimal"] = identify_pareto_front(scored, score_configs)
    threshold = max(1, int(np.ceil(scored.tp.max() * MIN_TP_RATIO)))
    scored["robust_candidate"] = scored.tp >= threshold
    return scored


def select_metric_equilibrium_row(data, config, preferred_iou=REFERENCE_IOU):
    metric, goal = config["column"], config["goal"]
    grouped_rows = []
    for _, subset in data.groupby("iou"):
        index = subset[metric].idxmax() if goal == "max" else subset[metric].idxmin()
        grouped_rows.append(data.loc[index])
    candidates = pd.DataFrame(grouped_rows).copy()
    best = candidates[metric].max() if goal == "max" else candidates[metric].min()
    near_best = candidates[
        candidates[metric] >= best - METRIC_TOLERANCE
        if goal == "max" else candidates[metric] <= best + METRIC_TOLERANCE
    ].copy()
    near_best["iou_distance"] = (near_best.iou - preferred_iou).abs()
    return near_best.sort_values(
        ["iou_distance", "weighted_score", "wins", "tp", "iou"],
        ascending=[True, False, False, False, False],
    ).iloc[0]


def collect_metric_best_rows(data, metrics_config):
    rows = []
    for config in metrics_config:
        equilibrium = select_metric_equilibrium_row(data, config)
        rows.append({
            "metric": config["label"], "column": config["column"],
            "goal": config["goal"], "equilibrium_iou": equilibrium.iou,
            "equilibrium_confidence": equilibrium.confidence,
            "equilibrium_weighted_score": equilibrium.weighted_score,
            "equilibrium_metric_value": equilibrium[config["column"]],
            "used_for_opt": config["label"] in OPT_SCORE_LABELS,
        })
    return rows


def get_metric_row(spec, metric_label):
    return next((item for item in spec["metric_best_rows"]
                 if item["metric"] == metric_label), None)


def build_analysis_specs(source: pd.DataFrame):
    data = prepare_metrics(source)
    specs = []
    for class_name in sorted(data["class"].dropna().unique()):
        scored = score_configurations(data[data["class"] == class_name].copy(),
                                      CLASS_METRICS_CONFIG)
        specs.append({"title": class_name, "scored": scored,
                      "metric_best_rows": collect_metric_best_rows(
                          scored, CLASS_METRICS_CONFIG), "type": "class"})
    general = score_configurations(build_general_dataframe(data), GENERAL_METRICS_CONFIG)
    specs.append({"title": "General", "scored": general,
                  "metric_best_rows": collect_metric_best_rows(
                      general, GENERAL_METRICS_CONFIG), "type": "general"})

    best_rows, selected_rows = [], []
    for spec in specs:
        for item in spec["metric_best_rows"]:
            best_rows.append({
                "graph": spec["title"], "metric": item["metric"],
                "goal": item["goal"], "optimal_iou": item["equilibrium_iou"],
                "optimal_confidence": item["equilibrium_confidence"],
                "optimal_metric_value": item["equilibrium_metric_value"],
                "score_at_optimal": item["equilibrium_weighted_score"],
            })
        f1_row = get_metric_row(spec, "F1")
        selected = spec["scored"][
            np.isclose(spec["scored"].iou, f1_row["equilibrium_iou"])
            & np.isclose(spec["scored"].confidence, f1_row["equilibrium_confidence"])
        ].iloc[0]
        class_specific = spec["type"] == "class"
        selected_rows.append({
            "graph": spec["title"], "selected_by": "F1",
            "iou": f1_row["equilibrium_iou"],
            "confidence": f1_row["equilibrium_confidence"],
            "f1_value": selected["cl_f1" if class_specific else "f1"],
            "ap50_at_f1": selected["cl_ap50" if class_specific else "map50"],
            "precision_at_f1": selected["cl_p" if class_specific else "precision"],
            "recall_at_f1": selected["cl_r" if class_specific else "recall"],
            "tp_at_f1": selected.tp, "fp_at_f1": selected.fp, "fn_at_f1": selected.fn,
        })
    return specs, pd.DataFrame(best_rows), pd.DataFrame(selected_rows)


def _heatmap(ax, spec, metric_label, *, large=False):
    column = ({"AP50": "cl_ap50", "F1": "cl_f1"}[metric_label]
              if spec["type"] == "class" else {"AP50": "map50", "F1": "f1"}[metric_label])
    pivot = spec["scored"].pivot_table(
        index="iou", columns="confidence", values=column, aggfunc="mean").sort_index().sort_index(axis=1)
    image = ax.imshow(pivot.to_numpy(float), aspect="auto", origin="lower", cmap="YlGn")
    scale = 2.0 if large else 1.0
    if not large:
        ax.set_title(f"{spec['title']} - {metric_label}", fontsize=13)
    ax.set_xlabel("Confidence", fontsize=13 * scale)
    ax.set_ylabel("IoU", fontsize=13 * scale)
    confidences = list(pivot.columns)
    positions = (list(range(len(confidences))) if len(confidences) <= 7
                 else sorted(set(int(value) for value in np.linspace(0, len(confidences) - 1, 6))))
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{confidences[pos]:.2f}" for pos in positions])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])
    ax.tick_params(axis="both", labelsize=10 * scale)
    f1_row = get_metric_row(spec, "F1")
    conf_index = list(pivot.columns).index(f1_row["equilibrium_confidence"])
    iou_index = list(pivot.index).index(f1_row["equilibrium_iou"])
    ax.scatter(conf_index, iou_index, s=180 * scale, facecolors="none",
               edgecolors="black", linewidths=2 * scale)
    iou_label = (f"IoU: {f1_row['equilibrium_iou']:.2f}" if large
                 else f"IoU={f1_row['equilibrium_iou']:.2f}")
    ax.text(conf_index, iou_index - .18,
            f"{iou_label}\nConf={f1_row['equilibrium_confidence']:.2f}",
            ha="center", va="top", fontsize=10 * scale, color="black",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=.80))
    return image


def build_heatmap_figure(specs, metric_label, output_dir: Path = DEFAULT_FIGURE_DIR):
    """Generate 2D heatmaps for the requested metric."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    for ax, spec in zip(axes.flatten(), specs):
        image = _heatmap(ax, spec, metric_label)
        colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
        colorbar.set_label(metric_label, fontsize=13)
        colorbar.ax.tick_params(labelsize=10)
    title = "2D mAP50 heatmaps" if metric_label == "AP50" else "2D F1-score heatmaps"
    fig.suptitle(title, fontsize=20)
    slug = "ap50" if metric_label == "AP50" else "f1"
    fig.savefig(output_dir / f"heatmap_2d_{slug}.png", dpi=400,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if metric_label == "F1":
        with PdfPages(output_dir / "validation_f1_heatmaps.pdf") as pages:
            for spec in specs[:3]:
                page, ax = plt.subplots(figsize=(14.58, 9.58), constrained_layout=True)
                image = _heatmap(ax, spec, metric_label, large=True)
                colorbar = page.colorbar(image, ax=ax, fraction=.046, pad=.04)
                colorbar.set_label(metric_label, fontsize=26)
                colorbar.ax.tick_params(labelsize=20)
                page.savefig(
                    output_dir / f"heatmap_f1_{spec['title'].lower()}.png",
                    dpi=300, bbox_inches="tight", facecolor="white",
                )
                pages.savefig(page, bbox_inches="tight", facecolor="white")
                plt.close(page)


def _curve_style(ax, ylabel):
    ax.spines["top"].set_alpha(.2)
    ax.spines["right"].set_alpha(.2)
    ax.spines["top"].set_linewidth(.05)
    ax.spines["bottom"].set_linewidth(.5)
    ax.spines["left"].set_linewidth(.5)
    ax.spines["right"].set_linewidth(.05)
    ax.spines["left"].set_color("black")
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both", colors="black")
    ax.grid(axis="both", which="major", linewidth=.5, alpha=.6)
    ax.set_xlabel("Confidence", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim(left=0)


def build_sensitivity_curves(source: pd.DataFrame, output_dir: Path):
    """Generate the overlaid sensitivity curves by class and IoU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = source.copy()
    data["TPR"] = data.cl_r.replace([np.inf, -np.inf], 0).fillna(0)
    denominator = data.cl_p + data.cl_r
    data["f1_score"] = np.where(
        denominator > 0, 2 * data.cl_p * data.cl_r / denominator, 0.0)
    metrics = (("cl_ap50", "mAP50"), ("cl_p", "Precision"),
               ("f1_score", "F1-Score"), ("TPR", "TPR"))
    with PdfPages(output_dir / "sensitivity_curves.pdf") as pdf:
        for column, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(10, 7))
            for class_name in data.Class.dropna().unique():
                for index, iou in enumerate(LINESTYLE_MAP):
                    rows = data[(data.IoU == iou) & (data.Class == class_name)].sort_values("Confidence")
                    ax.plot(rows.Confidence, rows[column],
                            label=f"{class_name} IoU={iou}",
                            color=COLOR_MAP[class_name][index],
                            linestyle=LINESTYLE_MAP[iou], linewidth=2)
            _curve_style(ax, ylabel)
            fig.tight_layout()
            fig.savefig(output_dir / f"sensitivity_{column}.png", dpi=300,
                        bbox_inches="tight", transparent=True)
            pdf.savefig(fig, bbox_inches="tight", transparent=True)
            plt.close(fig)


def build_sensitivity_legend(output_dir: Path):
    """Generate the shared legend used by the four sensitivity panels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    for index, iou in enumerate(LINESTYLE_MAP):
        for class_name in COLOR_MAP:
            handles.append(Line2D(
                [0], [0], color=COLOR_MAP[class_name][index],
                linestyle=LINESTYLE_MAP[iou], linewidth=2,
                label=f"{class_name} IoU={iou}",
            ))
    fig, ax = plt.subplots(figsize=(20, 1.6))
    ax.axis("off")
    ax.legend(
        handles=handles, loc="center", ncol=4, frameon=False,
        fontsize=16, handlelength=2.3, columnspacing=2.6,
        handletextpad=.7, borderaxespad=0,
    )
    fig.savefig(output_dir / "sensitivity_legend.pdf", dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(output_dir / "sensitivity_legend.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def build_sensitivity_artifacts(source: pd.DataFrame, output_dir: Path):
    specs, metric_best, selected = build_analysis_specs(source)
    build_sensitivity_legend(output_dir)
    build_sensitivity_curves(source, output_dir)
    build_heatmap_figure(specs, "AP50", output_dir)
    build_heatmap_figure(specs, "F1", output_dir)
    prepared = prepare_metrics(source)
    crack = prepared[(prepared["class"] == "crack") & (prepared.iou == .7)
                     & prepared.confidence.isin([.23, .14, .11, .10, .08])].copy()
    return {"sensitivity_selected": selected,
            "sensitivity_metric_best": metric_best,
            "crack_thresholds": crack}


__all__ = ["build_analysis_specs", "build_heatmap_figure",
           "build_sensitivity_curves", "build_sensitivity_legend",
           "build_sensitivity_artifacts", "prepare_metrics"]
