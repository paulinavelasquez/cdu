"""Three-dimensional visualization of the CDU acquisition stages."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

from .strategies.cdu import (
    allocate_candidate_quotas,
    optimize_n_clusters,
    refine_cluster_candidates,
    retain_final_batch,
)


PROFILE_COLUMNS = (
    "crack_uncertainty",
    "patch_uncertainty",
    "pothole_uncertainty",
)
ROUND = 36
ACQUIRED_IMAGES = 720
FINAL_METHOD = "cdu_fm4_sl_opt055"
ACQUISITION_BUDGET = 20
CANDIDATE_FACTOR = 4
FINAL_ALPHA = 0.55
SNAPSHOT_SHA256 = "d0a2274aae48ca27cc379ab3a57a90e2dc957d684565b32a493cfd3e167c344b"


class _PrimitiveUnpickler(pickle.Unpickler):
    """Accept only the primitive types defined by the snapshot format."""

    def find_class(self, module, name):  # pragma: no cover - only for an invalid file
        raise pickle.UnpicklingError(
            f"The snapshot contains a disallowed object: {module}.{name}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_profile_snapshot(
    snapshot_path: Path | str,
) -> tuple[pd.DataFrame, dict]:
    """Read the profiles and reapply the official final-CDU algorithm."""

    snapshot_path = Path(snapshot_path).resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Uncertainty snapshot is missing: {snapshot_path}")

    digest = _sha256(snapshot_path)
    if digest != SNAPSHOT_SHA256:
        raise ValueError(
            "The snapshot does not match the artifact preserved for the 3D figure: "
            f"{digest}"
        )

    with snapshot_path.open("rb") as stream:
        snapshot = _PrimitiveUnpickler(stream).load()

    required = {
        "format", "profiles", "clusters", "dua_selected",
        "candidates", "selected", "metadata",
    }
    if not isinstance(snapshot, dict) or required - set(snapshot):
        raise ValueError("Invalid structure in the 3D visualization snapshot")
    if snapshot["format"] != "cdu-cluster-visualization-v1":
        raise ValueError("Unknown 3D visualization snapshot version")

    profiles = np.asarray(snapshot["profiles"], dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] != len(PROFILE_COLUMNS):
        raise ValueError(
            "Each image must have three uncertainty values: crack, patch, and pothole"
        )
    if len(profiles) < CANDIDATE_FACTOR * ACQUISITION_BUDGET:
        raise ValueError("The snapshot does not contain enough images for FM=4")
    if not np.isfinite(profiles).all() or (profiles < 0).any() or (profiles > 1).any():
        raise ValueError("Per-class uncertainties must be finite and lie in [0, 1]")

    row_count = len(profiles)
    cluster_labels = np.asarray(snapshot["clusters"], dtype=int)
    dua_selected = np.asarray(snapshot["dua_selected"], dtype=int)
    candidates = np.asarray(snapshot["candidates"], dtype=int)
    selected = np.asarray(snapshot["selected"], dtype=int)
    for name, values in (
        ("clusters", cluster_labels),
        ("dua_selected", dua_selected),
        ("candidates", candidates),
        ("selected", selected),
    ):
        if values.shape != (row_count,):
            raise ValueError(f"Invalid length in {name}")
    for name, values in (
        ("dua_selected", dua_selected),
        ("candidates", candidates),
        ("selected", selected),
    ):
        if not set(np.unique(values)) <= {0, 1}:
            raise ValueError(f"{name} must be a binary mask")
    if int(dua_selected.sum()) != 20:
        raise ValueError("The DUA reference must contain 20 images")
    if int(candidates.sum()) != 80 or int(selected.sum()) != 20:
        raise ValueError("The CDU snapshot must contain 80 candidates and 20 acquisitions")
    if not np.all(candidates[selected.astype(bool)]):
        raise ValueError("Every acquired image must belong to the candidate set")

    image_ids = tuple(f"profile_{index:06d}" for index in range(row_count))
    data = pd.DataFrame(profiles, columns=PROFILE_COLUMNS)
    data.insert(0, "image", image_ids)
    data["uncertainty"] = profiles.sum(axis=1)

    recorded = snapshot["metadata"]

    # The snapshot preserves the coordinates used by the visualization. Groups,
    # quotas, candidates, and acquisitions are recomputed by CDUFM4SLOpt055;
    # stored masks are used only to verify file integrity.
    official_labels, cluster_count, db_index = optimize_n_clusters(
        profiles,
        classes=len(PROFILE_COLUMNS),
    )
    if db_index is None:
        raise ValueError("The preserved profiles do not support a GMM partition")

    groups = [
        np.flatnonzero(official_labels == cluster_id).tolist()
        for cluster_id in sorted(set(official_labels))
    ]
    candidate_budget = min(CANDIDATE_FACTOR * ACQUISITION_BUDGET, row_count)
    quotas = allocate_candidate_quotas(
        [len(group) for group in groups],
        candidate_budget,
        rule="sainte_lague",
    )
    official_candidates: list[int] = []
    for group, quota in zip(groups, quotas):
        official_candidates.extend(
            refine_cluster_candidates(
                group=group,
                quota=quota,
                profiles=profiles,
                uncertainty=data["uncertainty"].to_numpy(float),
                alpha=FINAL_ALPHA,
            )
        )
    official_selected = retain_final_batch(
        candidates=official_candidates,
        uncertainty=data["uncertainty"].to_numpy(float),
        image_ids=image_ids,
        budget=ACQUISITION_BUDGET,
    )
    dua_images = _dua_selection(data)

    data["cluster"] = official_labels.astype(int)
    data["dua_selected"] = data.image.isin(dua_images)
    data["candidate"] = False
    data.loc[official_candidates, "candidate"] = True
    data["selected"] = False
    data.loc[official_selected, "selected"] = True

    metadata = {
        "method": FINAL_METHOD,
        "round": int(recorded["round"]),
        "acquired_images": int(recorded["acquired_images"]),
        "clusters": int(cluster_count),
        "db_index": float(db_index),
        "candidates": len(official_candidates),
        "factor": CANDIDATE_FACTOR,
        "alpha": FINAL_ALPHA,
        "quotas": quotas,
        "snapshot": snapshot_path,
        "snapshot_sha256": digest,
    }
    computed_db = davies_bouldin_score(profiles, official_labels)
    if not np.isclose(computed_db, metadata["db_index"]):
        raise ValueError("The recomputed DB index does not match the GMM partition")
    return data, metadata


def _load_round(cluster_csv: Path | str) -> tuple[pd.DataFrame, dict]:
    """Read profiles, groups, candidates, and selections recorded by a run."""

    cluster_csv = Path(cluster_csv).resolve()
    if not cluster_csv.is_file():
        raise FileNotFoundError(
            "The visualization requires selections/clusters_round_36.csv, "
            "generated by round 36 of the final CDU method."
        )

    selection_json = cluster_csv.with_name(f"round_{ROUND:02d}.json")
    if not selection_json.is_file():
        raise FileNotFoundError(f"Acquisition metadata is missing: {selection_json}")

    data = pd.read_csv(cluster_csv)
    required = {"image", "cluster", "candidate", "selected", *PROFILE_COLUMNS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "The round output does not identify the candidate set; missing "
            f"columns: {sorted(missing)}"
        )

    metadata = json.loads(selection_json.read_text())
    if metadata.get("method") != FINAL_METHOD or int(metadata.get("round", -1)) != ROUND:
        raise ValueError("The visualization accepts only round 36 of the final CDU method")

    for column in (*PROFILE_COLUMNS, "cluster", "candidate", "selected"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data.image.duplicated().any():
        raise ValueError("The round output contains duplicate identifiers")
    if not set(data.candidate.unique()) <= {0, 1}:
        raise ValueError("The candidate column must be binary")
    if not set(data.selected.unique()) <= {0, 1}:
        raise ValueError("The selected column must be binary")

    data["candidate"] = data.candidate.astype(bool)
    data["selected"] = data.selected.astype(bool)
    data["uncertainty"] = data[list(PROFILE_COLUMNS)].sum(axis=1)

    if int(data.candidate.sum()) != int(metadata["candidates"]):
        raise ValueError("The candidate count differs from the metadata")
    if int(data.selected.sum()) != ACQUISITION_BUDGET:
        raise ValueError("The final acquisition must contain exactly 20 images")
    if not data.loc[data.selected, "candidate"].all():
        raise ValueError("Every acquired image must belong to the candidate set")

    return data, metadata


def _fit_recorded_gmm(data: pd.DataFrame, clusters: int) -> GaussianMixture:
    """Reconstruct the GMM and confirm that it matches the recorded partition."""

    profiles = data[list(PROFILE_COLUMNS)].to_numpy(float)
    model = GaussianMixture(
        n_components=clusters,
        random_state=0,
        reg_covar=1e-6,
    ).fit(profiles)
    reconstructed = model.predict(profiles)
    agreement = adjusted_rand_score(data.cluster.astype(int), reconstructed)
    if not np.isclose(agreement, 1.0):
        raise ValueError(
            "The reconstructed GMM does not match the recorded groups "
            f"(ARI={agreement:.6f})"
        )
    return model


def _dua_selection(data: pd.DataFrame, budget: int = ACQUISITION_BUDGET) -> set[str]:
    """Apply the reference DUA scalar ranking to the same set."""

    ranked = data.sort_values(
        ["uncertainty", "image"],
        ascending=[True, True],
        kind="mergesort",
    )
    return set(ranked.tail(budget).image)


def _ellipsoid(model: GaussianMixture, component: int, points: int = 100):
    """Generate an ellipsoid wireframe for the 3D visualization."""

    covariance = model.covariances_[component]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    radii = 2.0 * np.sqrt(np.clip(eigenvalues, 0.0, None))

    u = np.linspace(0, 2 * np.pi, points)
    v = np.linspace(0, np.pi, points)
    sphere = np.stack(
        (
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ),
        axis=0,
    ).reshape(3, -1)
    transformed = eigenvectors @ np.diag(radii) @ sphere
    transformed += model.means_[component, :, None]
    shape = (points, points)
    return tuple(axis.reshape(shape) for axis in transformed)


def _group_summary(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Summarize capacity, quota, candidates, and acquisitions by group."""

    summary = (
        data.groupby("cluster", sort=True)
        .agg(
            pool_images=("image", "size"),
            candidates=("candidate", "sum"),
            acquired=("selected", "sum"),
        )
        .reset_index()
    )
    summary["quota"] = list(metadata["quotas"])
    return summary


def _plotly_layout(
    figure,
    data: pd.DataFrame,
    annotation: str,
    *,
    gmm_panel: bool = False,
) -> None:
    """Apply the interactive camera and 3D visualization style."""

    ranges = {}
    for axis_name, column in zip(("xaxis", "yaxis", "zaxis"), PROFILE_COLUMNS):
        values = data[column]
        margin = float(values.max() - values.min()) * 0.03
        ranges[axis_name] = [
            float(values.min()) - margin,
            float(values.max()) + margin,
        ]

    figure.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title="Crack",
                backgroundcolor="white",
                gridcolor="black",
                showbackground=True,
                range=ranges["xaxis"] if gmm_panel else None,
            ),
            yaxis=dict(
                title="Patch",
                backgroundcolor="white",
                gridcolor="black",
                showbackground=True,
                range=ranges["yaxis"] if gmm_panel else None,
            ),
            zaxis=dict(
                title="Pothole",
                backgroundcolor="white",
                gridcolor="black",
                showbackground=True,
                range=ranges["zaxis"] if gmm_panel else None,
            ),
            aspectmode="cube" if gmm_panel else "auto",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    figure.add_annotation(
        text=annotation.replace("\n", "<br>"),
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="black"),
        borderwidth=1,
        align="left",
    )


def _plotly_points(figure, subset: pd.DataFrame, color, size: int = 3) -> None:
    """Add 3D profiles with image identifiers and uncertainty on hover."""

    import plotly.graph_objects as go

    if subset.empty:
        return
    figure.add_trace(
        go.Scatter3d(
            x=subset[PROFILE_COLUMNS[0]],
            y=subset[PROFILE_COLUMNS[1]],
            z=subset[PROFILE_COLUMNS[2]],
            mode="markers",
            marker=dict(size=size, color=color),
            customdata=np.column_stack((subset.image, subset.uncertainty)),
            hovertemplate=(
                "image=%{customdata[0]}<br>"
                "crack=%{x:.4f}<br>patch=%{y:.4f}<br>pothole=%{z:.4f}<br>"
                "U(i)=%{customdata[1]:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )


def _plotly_ellipsoid(figure, model: GaussianMixture, component: int) -> None:
    """Add the 100 × 100 ellipsoid wireframe to the 3D visualization."""

    import plotly.graph_objects as go

    x, y, z = _ellipsoid(model, component, points=100)
    figure.add_trace(
        go.Scatter3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            opacity=0.2,
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _render_interactive_cluster_selection(
    data: pd.DataFrame,
    metadata: dict,
    output_directory: Path | str,
) -> dict:
    """Create four rotatable Plotly figures and their HTML files."""

    import plotly.express as px
    import plotly.graph_objects as go

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    clusters = int(metadata["clusters"])
    gmm = _fit_recorded_gmm(data, clusters)
    base_annotation = f"Iteration: {ROUND}\nLabeled images: {ACQUIRED_IMAGES}"
    cdu_annotation = (
        f"{base_annotation}\nCluster: {clusters}\n"
        f"Index DB: {float(metadata['db_index']):.2f}\n"
        f"FM: {int(metadata['factor'])}\nα: {float(metadata['alpha']):.2f}"
    )

    figures: dict[str, object] = {}

    dua = go.Figure()
    _plotly_points(
        dua,
        data,
        np.where(data.dua_selected, "#009E73", "#E0E0E0"),
    )
    _plotly_layout(dua, data, base_annotation)
    figures["dua"] = dua

    # Keep ``cluster`` numeric so Plotly applies its default continuous scale.
    # The G10 sequence retains the interactive visualization color scheme.
    gmm_figure = px.scatter_3d(
        data,
        x=PROFILE_COLUMNS[0],
        y=PROFILE_COLUMNS[1],
        z=PROFILE_COLUMNS[2],
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    gmm_figure.update_traces(marker=dict(size=3))
    gmm_figure.update_coloraxes(showscale=False)
    for component in range(clusters):
        _plotly_ellipsoid(gmm_figure, gmm, component)
    _plotly_layout(
        gmm_figure,
        data,
        f"{base_annotation}\nCluster: {clusters}\n"
        f"Index DB: {float(metadata['db_index']):.2f}",
        gmm_panel=True,
    )
    figures["gmm"] = gmm_figure

    candidates = go.Figure()
    candidate_colors = np.where(
        data.selected,
        "red",
        np.where(data.candidate, "darkred", "#E0E0E0"),
    )
    _plotly_points(candidates, data, candidate_colors)
    _plotly_layout(candidates, data, cdu_annotation)
    figures["candidates"] = candidates

    selected = go.Figure()
    _plotly_points(
        selected,
        data,
        np.where(data.selected, "red", "#E0E0E0"),
    )
    _plotly_layout(selected, data, cdu_annotation)
    figures["selected"] = selected

    filenames = {
        "dua": "Clustering_DUA.html",
        "gmm": "Clustering_GMM.html",
        "candidates": "Clustering_CDU_FM.html",
        "selected": "Clustering_CDU.html",
    }
    html_files: dict[str, Path] = {}
    static_files: dict[str, dict[str, Path]] = {}
    for name, figure in figures.items():
        destination = output_directory / filenames[name]
        figure.write_html(
            destination,
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
        )
        html_files[name] = destination
        stem = destination.with_suffix("")
        pdf = stem.with_suffix(".pdf")
        figure.write_image(pdf, format="pdf", width=400, height=400, scale=3)
        static_files[name] = {"pdf": pdf}

    return {"figures": figures, "html": html_files, "static": static_files}


def _plotly_artifacts(
    data: pd.DataFrame,
    metadata: dict,
    output_directory: Path | str,
) -> dict:
    """Collect static and interactive outputs derived from the same figures."""

    rendered = _render_interactive_cluster_selection(
        data,
        metadata,
        output_directory,
    )
    dua_images = tuple(sorted(data.loc[data.dua_selected, "image"]))
    return {
        "figures": rendered["static"],
        "interactive": rendered["figures"],
        "html": rendered["html"],
        "metadata": metadata,
        "groups": _group_summary(data, metadata),
        "dua_images": dua_images,
    }


def build_cluster_selection_figure(
    cluster_csv: Path | str,
    output_directory: Path | str,
) -> dict:
    """Build the figure from round 36 recorded by a benchmark run."""

    data, metadata = _load_round(cluster_csv)
    if "dua_selected" not in data:
        data["dua_selected"] = data.image.isin(_dua_selection(data))
    return _plotly_artifacts(data, metadata, output_directory)


def build_cluster_selection_visualizations(
    cluster_csv: Path | str,
    output_directory: Path | str,
) -> dict:
    """Regenerate static and interactive outputs from a benchmark run."""

    return build_cluster_selection_figure(cluster_csv, output_directory)


def build_cluster_snapshot_figure(
    snapshot_path: Path | str,
    output_directory: Path | str,
) -> dict:
    """Regenerate the four static panels from the preserved profiles."""

    data, metadata = load_profile_snapshot(snapshot_path)
    return _plotly_artifacts(data, metadata, output_directory)


def build_cluster_snapshot_visualizations(
    snapshot_path: Path | str,
    output_directory: Path | str,
) -> dict:
    """Regenerate PDFs and Plotly figures with the official calculation."""

    return build_cluster_snapshot_figure(snapshot_path, output_directory)


__all__ = [
    "build_cluster_selection_figure",
    "build_cluster_selection_visualizations",
    "build_cluster_snapshot_figure",
    "build_cluster_snapshot_visualizations",
    "load_profile_snapshot",
]
