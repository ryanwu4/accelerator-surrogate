#!/usr/bin/env python
"""Compare MeshGraphNet and conditional flow models on particle trajectories.

This utility loads a directory of ``*_particle_data.pt`` files, evaluates both
the MeshGraphNet checkpoint and the conditional flow checkpoint, and reports
per-sample beam matrix accuracy along with aggregate statistics and plots.

The script mirrors the existing training/inference pipelines so that
normalisation, graph construction, and conditioning behaviour stay aligned.
The structure leaves room for plugging in additional distributional metrics
without rewriting the evaluation loop.
"""

from __future__ import annotations

import argparse
import gc
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import torch

from accelerator_flow_model.train_norm_flow_conditional import ConditionalFlowModel
from src.graph_models.inference.run_mgn_inference import (
    TORCH_LOAD_OPTS,
    build_graph,
    compute_condition_features,
    denormalise,
    load_trajectory,
    load_settings,
    maybe_load_metadata,
    normalise_tensor,
    parse_hyperparams_from_checkpoint,
    subsample_particles,
)
from src.graph_models.models.graph_networks import MeshGraphNet


plt.rcParams.update({"figure.autolayout": True})


BEAM_EPS = 1e-25
EMITTANCE_EPS = 1e-25

BEAM_AXIS_LABELS = ["x", "px", "y", "py", "z", "pz"]

# Emittance planes expressed in canonical ordering: (x, px, y, py, z, pz)
EMITTANCE_PLANES = {
    "x_xp": (0, 1),
    "y_yp": (2, 3),
    "z_delta": (4, 5),
}
EMITTANCE_KEYS = tuple(list(EMITTANCE_PLANES.keys()) + ["fourd", "sixd"])
EMITTANCE_LABELS = {
    "x_xp": "eps_x",
    "y_yp": "eps_y",
    "z_delta": "eps_z",
    "fourd": "eps_4d",
    "sixd": "eps_6d",
}

CANONICAL_AXIS_ORDER = (0, 3, 1, 4, 2, 5)


@dataclass
class SampleMetrics:
    """Container for per-sample evaluation artefacts."""

    file_path: Path
    indices: np.ndarray
    true_final: np.ndarray
    flow_pred: np.ndarray
    mgn_pred: np.ndarray
    beam_true: np.ndarray
    beam_flow: np.ndarray
    beam_mgn: np.ndarray
    reference_pz: float
    emittance_true: Dict[str, float]
    emittance_flow: Dict[str, float]
    emittance_mgn: Dict[str, float]
    density_mse_flow: Dict[str, float]
    density_mse_mgn: Dict[str, float]


def compute_phase_space_density_mse(
    true_particles: np.ndarray,
    pred_particles: np.ndarray,
    bins: int = 50,
) -> Dict[str, float]:
    """Compute MSE between 2D histograms of phase space slices.

    Args:
        true_particles: (N, 6) array of true particle coordinates.
        pred_particles: (N, 6) array of predicted particle coordinates.
        bins: Number of bins for the 2D histogram.

    Returns:
        Dictionary mapping slice name (x_px, y_py, z_pz) to MSE value.
    """
    # Indices for (x, px), (y, py), (z, pz) in the raw (N, 6) array
    # Raw order is x, y, z, px, py, pz based on CANONICAL_AXIS_ORDER = (0, 3, 1, 4, 2, 5)
    pairs = {
        "x_px": (0, 3),
        "y_py": (1, 4),
        "z_pz": (2, 5),
    }

    mses = {}

    for name, (idx1, idx2) in pairs.items():
        t1, t2 = true_particles[:, idx1], true_particles[:, idx2]
        p1, p2 = pred_particles[:, idx1], pred_particles[:, idx2]

        # Determine common range
        min1 = min(t1.min(), p1.min())
        max1 = max(t1.max(), p1.max())
        min2 = min(t2.min(), p2.min())
        max2 = max(t2.max(), p2.max())

        # Compute histograms
        H_true, _, _ = np.histogram2d(
            t1, t2, bins=bins, range=[[min1, max1], [min2, max2]], density=True
        )
        H_pred, _, _ = np.histogram2d(
            p1, p2, bins=bins, range=[[min1, max1], [min2, max2]], density=True
        )

        mse = np.mean((H_true - H_pred) ** 2)
        mses[name] = float(mse)

    return mses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MeshGraphNet and conditional flow models"
    )
    parser.add_argument(
        "--particle-dir",
        type=Path,
        required=True,
        help="Directory with *_particle_data.pt files",
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        required=True,
        help="Conditional flow checkpoint (.pt)",
    )
    parser.add_argument(
        "--flow-scalers", type=Path, required=True, help="Pickle with flow scalers"
    )
    parser.add_argument(
        "--mgn-checkpoint",
        type=Path,
        required=True,
        help="MeshGraphNet checkpoint (.pth)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Metadata JSON with normalisation + edge config",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Optional accelerator settings tensor (.pt)",
    )
    parser.add_argument(
        "--settings-dim",
        type=int,
        default=None,
        help="Expected accelerator settings dimension",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate at most N particle files"
    )
    parser.add_argument(
        "--n-particles", type=int, default=2000, help="Particles to subsample per file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="[Deprecated] Override to apply the same seed to both subsampling and flow sampling",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=63,
        help="Seed for particle subsampling (defaults to MeshGraphNet training seed)",
    )
    parser.add_argument(
        "--flow-seed",
        type=int,
        default=42,
        help="Seed for conditional flow sampling (defaults to flow training seed)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Target device: auto|cpu|cuda|mps"
    )
    parser.add_argument(
        "--edge-attr-method",
        type=str,
        default=None,
        help="Override edge attr method if needed",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Override k-NN value for graph construction"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None, help="Fallback MeshGraphNet hidden dim"
    )
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Fallback MeshGraphNet layer count"
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=5,
        help="How many samples to plot in detail",
    )
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Persist per-sample beam-matrix metrics to CSV",
    )
    return parser.parse_args()


def resolve_device(device_spec: str) -> torch.device:
    device_spec = device_spec.lower()
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    if device_spec == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device("cuda")
    if device_spec == "mps":
        if not torch.backends.mps.is_available():  # type: ignore[attr-defined]
            raise ValueError("MPS requested but not available")
        return torch.device("mps")
    if device_spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unrecognised device spec: {device_spec}")


def load_flow_model(
    checkpoint_path: Path,
    scalers_path: Path,
    device: torch.device,
) -> Tuple[ConditionalFlowModel, object, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", **TORCH_LOAD_OPTS)
    model_cfg = checkpoint.get("model_config")
    if model_cfg is None:
        raise RuntimeError("Conditional flow checkpoint missing model_config")
    # Older checkpoints stored the hidden dimension under ``hidden_units``.
    # Normalize to the constructor keyword expected by ``ConditionalFlowModel``.
    if "hidden_units" in model_cfg and "hidden_dim" not in model_cfg:
        model_cfg = dict(model_cfg)
        model_cfg["hidden_dim"] = model_cfg.pop("hidden_units")
    model = ConditionalFlowModel(**model_cfg)
    state_dict = checkpoint.get("flow_state_dict")
    if state_dict is None:
        raise RuntimeError("Conditional flow checkpoint missing flow_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with open(scalers_path, "rb") as handle:
        scalers = pickle.load(handle)
    scaler_final = scalers.get("scaler_final")
    scaler_condition = scalers.get("scaler_condition")
    if scaler_final is None or scaler_condition is None:
        raise RuntimeError("Scalers pickle missing expected entries")

    return model, scaler_final, scaler_condition


def infer_settings_dim(settings_path: Optional[Path]) -> int:
    if settings_path is None:
        raise ValueError(
            "settings_dim must be provided when no settings tensor is supplied"
        )
    payload = torch.load(settings_path, **TORCH_LOAD_OPTS)
    if isinstance(payload, dict):
        from src.preprocessing.generate_graphs_from_sequence_particles import (
            settings_dict_to_tensor,
        )

        tensor = settings_dict_to_tensor(payload)
        return int(tensor.numel())
    tensor = torch.as_tensor(payload)
    return int(tensor.numel())


def list_particle_files(particle_dir: Path) -> List[Path]:
    return sorted(p for p in particle_dir.glob("*_particle_data.pt") if p.is_file())

def final_to_beam_coordinates(
    array: np.ndarray, reference_pz: Optional[float] = None
) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"Expected (N,6) array, received {array.shape}")
    if array.shape[0] < 2:
        raise ValueError("Need at least two particles to compute beam coordinates")

    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    px = array[:, 3]
    py = array[:, 4]
    pz = array[:, 5]
    return np.stack((x, px, y, py, z, pz), axis=1).astype(np.float32)

def compute_beam_matrix(
    array: np.ndarray, reference_pz: Optional[float] = None
) -> np.ndarray:
    beam_coords = final_to_beam_coordinates(array, reference_pz=reference_pz)
    return np.cov(beam_coords, rowvar=False)


def to_canonical_coordinates(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"Expected (N,6) array, received {array.shape}")
    return array[:, CANONICAL_AXIS_ORDER].astype(np.float32)


def _safe_det(matrix: np.ndarray) -> float:
    det = float(np.linalg.det(matrix))
    if det < 0 and abs(det) < 1e-12:
        det = 0.0
    if det < 0:
        raise ValueError("Beam matrix determinant negative beyond tolerance")
    return det


def _moment_emittance(coord_a: np.ndarray, coord_b: np.ndarray) -> float:
    mean_a2 = float(np.mean(coord_a**2))
    mean_b2 = float(np.mean(coord_b**2))
    mean_ab = float(np.mean(coord_a * coord_b))
    area = mean_a2 * mean_b2 - mean_ab**2
    if area < 0 and abs(area) < 1e-16:
        area = 0.0
    if area < 0:
        raise ValueError("Emittance area negative beyond tolerance")
    return float(np.sqrt(area))


def compute_emittances(
    canonical_coords: np.ndarray,
    canonical_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if canonical_coords.ndim != 2 or canonical_coords.shape[1] != 6:
        raise ValueError(
            f"Expected canonical coordinates with shape (N,6), received {canonical_coords.shape}"
        )
    emittances: Dict[str, float] = {}
    for name, (i, j) in EMITTANCE_PLANES.items():
        emittances[name] = _moment_emittance(
            canonical_coords[:, i], canonical_coords[:, j]
        )

    canonical = canonical_matrix
    if canonical is None:
        canonical = np.cov(canonical_coords, rowvar=False)

    fourd_indices = (0, 1, 2, 3)
    emittances["fourd"] = float(
        np.sqrt(_safe_det(canonical[np.ix_(fourd_indices, fourd_indices)]))
    )
    emittances["sixd"] = float(np.sqrt(_safe_det(canonical)))
    return emittances


def _add_covariance_ellipse(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
    label: Optional[str] = None,
) -> None:
    if xs.size < 2 or ys.size < 2:
        return
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        return
    cov = np.cov(np.vstack((xs, ys)))
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals <= 0):
        return
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width, height = 2.0 * np.sqrt(eigvals)
    if not np.isfinite(width) or not np.isfinite(height):
        return
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ellipse = Ellipse(
        (float(np.mean(xs)), float(np.mean(ys))),
        width=float(width),
        height=float(height),
        angle=float(angle),
        edgecolor=color,
        facecolor=color,
        linewidth=1.0,
        linestyle="-",
        alpha=0.4,
        label=label,
    )
    ax.add_patch(ellipse)


def sanitise_particle_pair(
    initial: np.ndarray,
    final: np.ndarray,
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if initial.shape[0] != final.shape[0]:
        raise ValueError(
            f"Initial ({initial.shape[0]}) and final ({final.shape[0]}) particle counts differ for {file_path.name}"
        )
    if not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise ValueError(
            f"NaN/Inf detected; dropping entire bunch for {file_path.name}"
        )
    indices = np.arange(initial.shape[0])
    return initial, final, indices


def sample_conditional_flow(
    model: ConditionalFlowModel,
    scaler_final,
    scaler_condition,
    initial_particles: np.ndarray,
    n_particles: int,
    device: torch.device,
    torch_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    condition_stats = compute_condition_features(initial_particles)
    cond_norm = scaler_condition.transform(condition_stats.reshape(1, -1)).astype(
        np.float32
    )
    cond_tensor = torch.from_numpy(cond_norm).to(device)
    cond_expanded = cond_tensor.expand(n_particles, -1)

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    with torch.no_grad():
        if device.type == "mps":
            model_cpu = model.to("cpu")
            samples_norm = model_cpu.sample(n_particles, cond_expanded.cpu())
            model.to(device)
        else:
            samples_norm = model.sample(n_particles, cond_expanded)
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)

    samples_np = samples_norm.detach().cpu().numpy()
    samples = scaler_final.inverse_transform(samples_np)
    return samples.astype(np.float32), condition_stats.astype(np.float32)


def initialise_mgn_model(
    graph,
    checkpoint_path: Path,
    hidden_dim_override: Optional[int],
    num_layers_override: Optional[int],
    device: torch.device,
) -> MeshGraphNet:
    hyper = parse_hyperparams_from_checkpoint(checkpoint_path)
    hidden_dim = hidden_dim_override or (
        int(hyper["hidden_dim"]) if hyper["hidden_dim"] else None
    )
    num_layers = num_layers_override or (
        int(hyper["num_layers"]) if hyper["num_layers"] else None
    )
    if hidden_dim is None:
        raise ValueError(
            "Unable to resolve MeshGraphNet hidden_dim; provide --hidden-dim"
        )
    if num_layers is None:
        raise ValueError(
            "Unable to resolve MeshGraphNet num_layers; provide --num-layers"
        )

    node_in_dim = graph.x.shape[1]
    edge_in_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else 0
    model = MeshGraphNet(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        node_out_dim=6,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", **TORCH_LOAD_OPTS)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("MeshGraphNet checkpoint missing model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model._node_in_dim = node_in_dim  # type: ignore[attr-defined]
    model._edge_in_dim = edge_in_dim  # type: ignore[attr-defined]
    return model


def run_mgn_prediction(
    model: MeshGraphNet,
    initial_raw: torch.Tensor,
    settings: torch.Tensor,
    edge_config: Dict[str, object],
    edge_attr_method: Optional[str],
    norm_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> np.ndarray:
    if norm_stats is not None:
        initial_norm = normalise_tensor(initial_raw, norm_stats["initial"])
    else:
        initial_norm = initial_raw.clone()

    graph = build_graph(initial_norm, settings, edge_config, edge_attr_method)
    graph = graph.to(device)

    with torch.no_grad():
        pred_norm = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

    final_stats = norm_stats["final"] if norm_stats is not None else None
    pred_denorm = denormalise(pred_norm.cpu(), final_stats)
    return pred_denorm.numpy().astype(np.float32)


def prepare_edge_config(
    metadata_bundle: Dict[str, object],
    k_override: Optional[int],
) -> Tuple[Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], Dict[str, object]]:
    norm_stats = metadata_bundle["stats"] if "stats" in metadata_bundle else None
    edge_config = dict(metadata_bundle.get("edge_config", {}))
    if k_override is not None:
        edge_config["k"] = k_override
    return norm_stats, edge_config


def evaluate_sample(
    file_path: Path,
    initial_sub: np.ndarray,
    final_sub: np.ndarray,
    indices: np.ndarray,
    flow_seed: int,
    flow_model: ConditionalFlowModel,
    scaler_final,
    scaler_condition,
    mgn_model: MeshGraphNet,
    settings: torch.Tensor,
    norm_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    edge_config: Dict[str, object],
    edge_attr_method: Optional[str],
    device: torch.device,
) -> SampleMetrics:
    flow_pred, _ = sample_conditional_flow(
        model=flow_model,
        scaler_final=scaler_final,
        scaler_condition=scaler_condition,
        initial_particles=initial_sub,
        n_particles=len(indices),
        device=device,
        torch_seed=flow_seed,
    )

    initial_raw = torch.from_numpy(initial_sub.astype(np.float32))
    mgn_pred = run_mgn_prediction(
        model=mgn_model,
        initial_raw=initial_raw,
        settings=settings,
        edge_config=dict(edge_config),
        edge_attr_method=edge_attr_method,
        norm_stats=norm_stats,
        device=device,
    )

    if not np.isfinite(flow_pred).all():
        raise ValueError("Flow prediction produced NaN or Inf values")
    if not np.isfinite(mgn_pred).all():
        raise ValueError("MeshGraphNet prediction produced NaN or Inf values")
    if not np.isfinite(final_sub).all():
        raise ValueError(
            "True final particles contain NaN or Inf values after sanitisation"
        )

    reference_pz = float(np.mean(final_sub[:, 5]))
    final_true = final_sub.astype(np.float32)
    beam_true = compute_beam_matrix(final_true, reference_pz=reference_pz)
    beam_flow = compute_beam_matrix(flow_pred, reference_pz=reference_pz)
    beam_mgn = compute_beam_matrix(mgn_pred, reference_pz=reference_pz)

    canonical_true_coords = to_canonical_coordinates(final_true)
    canonical_flow_coords = to_canonical_coordinates(flow_pred)
    canonical_mgn_coords = to_canonical_coordinates(mgn_pred)

    emittance_true = compute_emittances(canonical_true_coords)
    emittance_flow = compute_emittances(canonical_flow_coords)
    emittance_mgn = compute_emittances(canonical_mgn_coords)

    density_mse_flow = compute_phase_space_density_mse(final_true, flow_pred)
    density_mse_mgn = compute_phase_space_density_mse(final_true, mgn_pred)

    return SampleMetrics(
        file_path=file_path,
        indices=indices,
        true_final=final_true,
        flow_pred=flow_pred,
        mgn_pred=mgn_pred,
        beam_true=beam_true,
        beam_flow=beam_flow,
        beam_mgn=beam_mgn,
        reference_pz=reference_pz,
        emittance_true=emittance_true,
        emittance_flow=emittance_flow,
        emittance_mgn=emittance_mgn,
        density_mse_flow=density_mse_flow,
        density_mse_mgn=density_mse_mgn,
    )


def stack_beam_matrices(
    samples: Sequence[SampleMetrics],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beam_true = np.stack([s.beam_true for s in samples])
    beam_flow = np.stack([s.beam_flow for s in samples])
    beam_mgn = np.stack([s.beam_mgn for s in samples])
    return beam_true, beam_flow, beam_mgn


def beam_summary(
    beam_true: np.ndarray,
    beam_flow: np.ndarray,
    beam_mgn: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    diffs_flow = beam_flow - beam_true
    diffs_mgn = beam_mgn - beam_true

    baseline_abs = np.mean(np.abs(beam_true), axis=0)
    baseline_abs = np.where(baseline_abs < BEAM_EPS, BEAM_EPS, baseline_abs)

    records = []
    for i in range(6):
        for j in range(6):
            flow_abs = np.abs(diffs_flow[:, i, j])
            mgn_abs = np.abs(diffs_mgn[:, i, j])
            flow_pct = flow_abs / baseline_abs[i, j] * 100.0
            mgn_pct = mgn_abs / baseline_abs[i, j] * 100.0
            flow_signed_pct = diffs_flow[:, i, j] / baseline_abs[i, j] * 100.0
            mgn_signed_pct = diffs_mgn[:, i, j] / baseline_abs[i, j] * 100.0
            records.append(
                {
                    "i": i,
                    "j": j,
                    "model": "flow",
                    "mean_abs_error": float(flow_abs.mean()),
                    "std_abs_error": float(flow_abs.std()),
                    "median_abs_error": float(np.median(flow_abs)),
                    "mean_signed_error": float(diffs_flow[:, i, j].mean()),
                    "mean_abs_percent_error": float(flow_pct.mean()),
                    "std_abs_percent_error": float(flow_pct.std()),
                    "median_abs_percent_error": float(np.median(flow_pct)),
                    "mean_signed_percent_error": float(flow_signed_pct.mean()),
                    "baseline_abs_mean": float(baseline_abs[i, j]),
                }
            )
            records.append(
                {
                    "i": i,
                    "j": j,
                    "model": "mgn",
                    "mean_abs_error": float(mgn_abs.mean()),
                    "std_abs_error": float(mgn_abs.std()),
                    "median_abs_error": float(np.median(mgn_abs)),
                    "mean_signed_error": float(diffs_mgn[:, i, j].mean()),
                    "mean_abs_percent_error": float(mgn_pct.mean()),
                    "std_abs_percent_error": float(mgn_pct.std()),
                    "median_abs_percent_error": float(np.median(mgn_pct)),
                    "mean_signed_percent_error": float(mgn_signed_pct.mean()),
                    "baseline_abs_mean": float(baseline_abs[i, j]),
                }
            )
    summary = pd.DataFrame.from_records(records)
    return summary, baseline_abs


def emittance_summary(samples: Sequence[SampleMetrics]) -> pd.DataFrame:
    records = []
    for plane in EMITTANCE_KEYS:
        true_vals = np.array([s.emittance_true[plane] for s in samples])
        flow_vals = np.array([s.emittance_flow[plane] for s in samples])
        mgn_vals = np.array([s.emittance_mgn[plane] for s in samples])

        baseline = np.where(
            np.abs(true_vals) < EMITTANCE_EPS, EMITTANCE_EPS, np.abs(true_vals)
        )

        for model, vals in (("flow", flow_vals), ("mgn", mgn_vals)):
            diffs = vals - true_vals
            abs_diffs = np.abs(diffs)
            pct_abs = abs_diffs / baseline * 100.0
            pct_signed = diffs / baseline * 100.0
            records.append(
                {
                    "plane": plane,
                    "model": model,
                    "mean_abs_error": float(abs_diffs.mean()),
                    "std_abs_error": float(abs_diffs.std()),
                    "median_abs_error": float(np.median(abs_diffs)),
                    "mean_signed_error": float(diffs.mean()),
                    "mean_abs_percent_error": float(pct_abs.mean()),
                    "std_abs_percent_error": float(pct_abs.std()),
                    "median_abs_percent_error": float(np.median(pct_abs)),
                    "mean_signed_percent_error": float(pct_signed.mean()),
                    "baseline_abs_mean": float(baseline.mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_beam_heatmaps(
    df: pd.DataFrame,
    output_dir: Path,
    value_col: str,
    filename_prefix: str,
    title_prefix: str,
    colorbar_label: str,
    formatter,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        heat = np.zeros((6, 6))
        for _, row in subset.iterrows():
            heat[int(row["i"]), int(row["j"])] = row[value_col]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(heat, cmap="viridis")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(BEAM_AXIS_LABELS)
        ax.set_yticklabels(BEAM_AXIS_LABELS)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(f"{title_prefix} ({model.upper()})")
        for i in range(6):
            for j in range(6):
                ax.text(
                    j,
                    i,
                    formatter(heat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        fig.savefig(output_dir / f"{filename_prefix}_{model}.png", dpi=200)
        plt.close(fig)


def plot_beam_line(
    df: pd.DataFrame,
    output_path: Path,
    value_col: str,
    ylabel: str,
    title: str,
    as_percent: bool = False,
) -> None:
    pivot = (
        df.pivot_table(index=["i", "j"], columns="model", values=value_col)
        .reset_index()
        .sort_values(by=["i", "j"])
    )
    labels = [f"({int(row.i)},{int(row.j)})" for _, row in pivot.iterrows()]
    fig, ax = plt.subplots(figsize=(14, 4))
    available_models = [m for m in ("flow", "mgn") if m in pivot.columns]
    if not available_models:
        print(
            f"  Skipping {output_path.name}: no models with column {value_col} present in summary"
        )
        plt.close(fig)
        return
    for model in available_models:
        ax.plot(labels, pivot[model], label=model.upper(), marker="o")
    ax.set_xlabel("Beam matrix index (row,col)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend()
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _format_emittance_annotation(sample: SampleMetrics) -> str:
    header = "metric    true         Flow pred (|err|, %)      MGN pred (|err|, %)"
    divider = "-" * len(header)
    rows: List[str] = [header, divider]
    for plane in ("x_xp", "y_yp", "z_delta", "fourd", "sixd"):
        label = EMITTANCE_LABELS[plane]
        true_val = sample.emittance_true[plane]
        baseline = max(abs(true_val), EMITTANCE_EPS)

        flow_pred = sample.emittance_flow[plane]
        flow_err = abs(flow_pred - true_val)
        flow_pct = flow_err / baseline * 100.0 if baseline > 0 else 0.0

        mgn_pred = sample.emittance_mgn[plane]
        mgn_err = abs(mgn_pred - true_val)
        mgn_pct = mgn_err / baseline * 100.0 if baseline > 0 else 0.0

        rows.append(
            f"{label:<8}{true_val:>11.4e}  {flow_pred:>11.4e} ({flow_err:>7.2e}, {flow_pct:5.2f}%)  "
            f"{mgn_pred:>11.4e} ({mgn_err:>7.2e}, {mgn_pct:5.2f}%)"
        )
    return "\n".join(rows)


def plot_representative_samples(
    samples: Sequence[SampleMetrics],
    output_dir: Path,
    annotate_emittance: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    coord_specs = {
        0: ("x", lambda arr: arr * 1e3, "x [mm]"),
        1: ("y", lambda arr: arr * 1e3, "y [mm]"),
        2: ("z", lambda arr: arr * 1e3, "z [mm]"),
        3: ("px", lambda arr: arr, "px [eV/c]"),
        4: ("py", lambda arr: arr, "py [eV/c]"),
        5: ("pz", lambda arr: arr / 1e6, "pz [MeV/c]"),
    }

    for idx, sample in enumerate(samples, start=1):
        fig_hist, axes_hist = plt.subplots(2, 3, figsize=(14, 8))
        for dim, ax in enumerate(axes_hist.flatten()):
            _, transform, label = coord_specs[dim]
            bins = 60
            ax.hist(
                transform(sample.true_final[:, dim]),
                bins=bins,
                alpha=0.5,
                label="True",
                density=True,
                color="steelblue",
            )
            ax.hist(
                transform(sample.flow_pred[:, dim]),
                bins=bins,
                alpha=0.45,
                label="Flow",
                density=True,
                color="firebrick",
            )
            ax.hist(
                transform(sample.mgn_pred[:, dim]),
                bins=bins,
                alpha=0.45,
                label="MGN",
                density=True,
                color="forestgreen",
            )
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)
            if dim == 0:
                ax.legend()
        fig_hist.suptitle(
            f"Marginal distributions – sample {idx}: {sample.file_path.name}"
        )
        fig_hist.tight_layout(rect=(0, 0.26, 1, 1))
        if annotate_emittance:
            fig_hist.text(
                0.5,
                0.03,
                _format_emittance_annotation(sample),
                ha="center",
                va="bottom",
                fontsize=7.5,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.75,
                ),
            )
        fig_hist.savefig(output_dir / f"sample_{idx:02d}_marginals.png", dpi=200)
        plt.close(fig_hist)

        fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(15, 5))
        scatter_specs = [
            (0, 3),
            (1, 4),
            (2, 5),
        ]
        labels_for_ellipses = {"True": "True ellipse", "Flow": "Flow ellipse", "MGN": "MGN ellipse"}
        for ax, (i_dim, j_dim) in zip(axes_scatter, scatter_specs):
            _, transform_i, label_i = coord_specs[i_dim]
            _, transform_j, label_j = coord_specs[j_dim]
            ax.scatter(
                transform_i(sample.true_final[:, i_dim]),
                transform_j(sample.true_final[:, j_dim]),
                s=4,
                alpha=0.35,
                label="True",
                color="steelblue",
            )
            ax.scatter(
                transform_i(sample.flow_pred[:, i_dim]),
                transform_j(sample.flow_pred[:, j_dim]),
                s=4,
                alpha=0.35,
                label="Flow",
                color="firebrick",
            )
            ax.scatter(
                transform_i(sample.mgn_pred[:, i_dim]),
                transform_j(sample.mgn_pred[:, j_dim]),
                s=4,
                alpha=0.35,
                label="MGN",
                color="forestgreen",
            )
            ax.set_xlabel(label_i)
            ax.set_ylabel(label_j)
            ax.grid(alpha=0.3)
            true_x = transform_i(sample.true_final[:, i_dim])
            true_y = transform_j(sample.true_final[:, j_dim])
            flow_x = transform_i(sample.flow_pred[:, i_dim])
            flow_y = transform_j(sample.flow_pred[:, j_dim])
            mgn_x = transform_i(sample.mgn_pred[:, i_dim])
            mgn_y = transform_j(sample.mgn_pred[:, j_dim])
            _add_covariance_ellipse(ax, true_x, true_y, color="steelblue", label=labels_for_ellipses.get("True"))
            _add_covariance_ellipse(ax, flow_x, flow_y, color="firebrick", label=labels_for_ellipses.get("Flow"))
            _add_covariance_ellipse(ax, mgn_x, mgn_y, color="forestgreen", label=labels_for_ellipses.get("MGN"))
            labels_for_ellipses = {"True": None, "Flow": None, "MGN": None}
        axes_scatter[0].legend()
        fig_scatter.suptitle(
            f"Phase-space overlays – sample {idx}: {sample.file_path.name}"
        )
        fig_scatter.tight_layout(rect=(0, 0.30, 1, 1))
        if annotate_emittance:
            fig_scatter.text(
                0.5,
                0.03,
                _format_emittance_annotation(sample),
                ha="center",
                va="bottom",
                fontsize=7.5,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.75,
                ),
            )
        fig_scatter.savefig(
            output_dir / f"sample_{idx:02d}_phase_overlays.png", dpi=200
        )
        plt.close(fig_scatter)


def plot_beam_entry_distributions(
    samples: Sequence[SampleMetrics], output_dir: Path, bins: int = 50
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    coord_specs = {
        0: ("x", 1e3, "x [mm]"),
        1: ("px", 1.0, "px [eV/c]"),
        2: ("y", 1e3, "y [mm]"),
        3: ("py", 1.0, "py [eV/c]"),
        4: ("z", 1e3, "z [mm]"),
        5: ("pz", 1e-6, "pz [MeV/c]"),
    }
    for idx, sample in enumerate(samples, start=1):
        true_beam = final_to_beam_coordinates(
            sample.true_final, reference_pz=sample.reference_pz
        )
        flow_beam = final_to_beam_coordinates(
            sample.flow_pred, reference_pz=sample.reference_pz
        )
        mgn_beam = final_to_beam_coordinates(
            sample.mgn_pred, reference_pz=sample.reference_pz
        )

        true_centered = true_beam - true_beam.mean(axis=0, keepdims=True)
        flow_centered = flow_beam - flow_beam.mean(axis=0, keepdims=True)
        mgn_centered = mgn_beam - mgn_beam.mean(axis=0, keepdims=True)

        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        for i in range(6):
            for j in range(6):
                ax = axes[i, j]
                if j < i:
                    ax.axis("off")
                    continue
                scale_i = coord_specs[i][1]
                scale_j = coord_specs[j][1]
                label_i = coord_specs[i][2]
                label_j = coord_specs[j][2]
                if i == j:
                    for data, label, color in (
                        (true_centered[:, i] * scale_i, "True", "steelblue"),
                        (flow_centered[:, i] * scale_i, "Flow", "firebrick"),
                        (mgn_centered[:, i] * scale_i, "MGN", "forestgreen"),
                    ):
                        ax.hist(
                            data,
                            bins=bins,
                            alpha=0.35,
                            density=True,
                            color=color,
                            label=label if i == 0 else None,
                        )
                    ax.set_xlabel(label_i)
                    ax.set_ylabel("Density")
                else:
                    for x_vals, y_vals, label, color in (
                        (true_centered[:, j] * scale_j, true_centered[:, i] * scale_i, "True", "steelblue"),
                        (flow_centered[:, j] * scale_j, flow_centered[:, i] * scale_i, "Flow", "firebrick"),
                        (mgn_centered[:, j] * scale_j, mgn_centered[:, i] * scale_i, "MGN", "forestgreen"),
                    ):
                        ax.scatter(
                            x_vals,
                            y_vals,
                            s=4,
                            alpha=0.3,
                            label=label if (i == 0 and j == 1) else None,
                            color=color,
                        )
                    if i == 0:
                        ax.set_xlabel(label_j)
                    if j == 5:
                        ax.set_ylabel(label_i)
                ax.set_title(
                    f"({BEAM_AXIS_LABELS[i]},{BEAM_AXIS_LABELS[j]})", fontsize=8
                )
                ax.tick_params(axis="both", labelsize=6)
                ax.grid(alpha=0.2, linewidth=0.3)
        handles, labels = axes[0, 1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=9)
        fig.suptitle(
            f"Beam matrix entry distributions – sample {idx}: {sample.file_path.name}",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(
            output_dir / f"sample_{idx:02d}_beam_entry_distributions.png", dpi=200
        )
        plt.close(fig)


def select_top_emittance_samples(
    samples: Sequence[SampleMetrics],
    count: int = 5,
    model: str = "both",
) -> List[SampleMetrics]:
    """Select samples with highest 6D emittance errors.
    
    Args:
        samples: Sequence of sample metrics
        count: Number of top samples to return
        model: Which model to score by - "flow", "mgn", or "both" (max of both)
    
    Returns:
        List of top samples sorted by error (highest first)
    """
    if not samples:
        return []

    def score(sample: SampleMetrics) -> float:
        true_val = sample.emittance_true["sixd"]
        baseline = max(abs(true_val), EMITTANCE_EPS)
        flow_err = abs(sample.emittance_flow["sixd"] - true_val) / baseline * 100.0
        mgn_err = abs(sample.emittance_mgn["sixd"] - true_val) / baseline * 100.0
        
        if model == "flow":
            return flow_err
        elif model == "mgn":
            return mgn_err
        else:  # "both"
            return max(flow_err, mgn_err)

    ranked = sorted(samples, key=score, reverse=True)
    return ranked[: min(count, len(ranked))]


def plot_emittance_scatter(samples: Sequence[SampleMetrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colours = {"flow": "firebrick", "mgn": "forestgreen"}
    markers = {"flow": "o", "mgn": "^"}

    for plane in EMITTANCE_KEYS:
        true_vals = np.array([s.emittance_true[plane] for s in samples])
        fig, ax = plt.subplots(figsize=(5, 5))
        for model in ("flow", "mgn"):
            preds = np.array(
                [
                    (
                        s.emittance_flow[plane]
                        if model == "flow"
                        else s.emittance_mgn[plane]
                    )
                    for s in samples
                ]
            )
            ax.scatter(
                true_vals,
                preds,
                label=model.upper(),
                color=colours[model],
                marker=markers[model],
                alpha=0.75,
                s=36,
                edgecolor="black",
                linewidth=0.4,
            )
        combined = np.concatenate(
            [
                true_vals,
                np.array([s.emittance_flow[plane] for s in samples]),
                np.array([s.emittance_mgn[plane] for s in samples]),
            ]
        )
        min_val = float(np.min(combined))
        max_val = float(np.max(combined))
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            plt.close(fig)
            continue
        if np.isclose(min_val, max_val):
            span = abs(min_val) if abs(min_val) > 0 else 1.0
            min_plot = min_val - 0.05 * span
            max_plot = max_val + 0.05 * span
        else:
            pad = 0.05 * (max_val - min_val)
            min_plot = min_val - pad
            max_plot = max_val + pad
        ax.plot(
            [min_plot, max_plot],
            [min_plot, max_plot],
            linestyle="--",
            color="grey",
            linewidth=1,
        )
        ax.set_xlim(min_plot, max_plot)
        ax.set_ylim(min_plot, max_plot)
        ax.set_xlabel(f"True {EMITTANCE_LABELS[plane]}")
        ax.set_ylabel(f"Predicted {EMITTANCE_LABELS[plane]}")
        ax.set_title(f"Emittance scatter ({EMITTANCE_LABELS[plane]})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"emittance_{plane}_scatter.png", dpi=200)
        plt.close(fig)


def collect_emittance_metrics(samples: Sequence[SampleMetrics]) -> pd.DataFrame:
    records = []
    for sample in samples:
        for plane in EMITTANCE_KEYS:
            baseline = max(abs(sample.emittance_true[plane]), EMITTANCE_EPS)
            for model, value in (
                ("flow", sample.emittance_flow[plane]),
                ("mgn", sample.emittance_mgn[plane]),
            ):
                diff = value - sample.emittance_true[plane]
                records.append(
                    {
                        "file": sample.file_path.name,
                        "plane": plane,
                        "model": model,
                        "abs_error": float(abs(diff)),
                        "signed_error": float(diff),
                        "abs_percent_error": float(abs(diff) / baseline * 100.0),
                        "signed_percent_error": float(diff / baseline * 100.0),
                        "true_emittance": float(sample.emittance_true[plane]),
                        "pred_emittance": float(value),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_emittance_error_histograms(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    bins: int = 40,
) -> None:
    if metrics_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    colour_map = {"flow": "firebrick", "mgn": "forestgreen"}
    for value_col, suffix, label in (
        ("abs_error", "abs_error", "|error|"),
        ("abs_percent_error", "abs_percent_error", "|error| [%]"),
    ):
        fig, axes = plt.subplots(
            1, len(EMITTANCE_KEYS), figsize=(4 * len(EMITTANCE_KEYS), 4)
        )
        if len(EMITTANCE_KEYS) == 1:
            axes = np.array([axes])
        for ax, plane in zip(axes.flat, EMITTANCE_KEYS):
            plane_df = metrics_df[metrics_df["plane"] == plane]
            for model, colour in colour_map.items():
                values = plane_df[plane_df["model"] == model][value_col].to_numpy()
                if values.size == 0:
                    continue
                ax.hist(
                    values,
                    bins=bins,
                    alpha=0.55,
                    density=True,
                    color=colour,
                    label=model.upper() if plane == EMITTANCE_KEYS[0] else None,
                )
            ax.set_title(EMITTANCE_LABELS[plane])
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"emittance_{suffix}_histograms.png", dpi=200)
        plt.close(fig)


def plot_density_mse_distributions(
    samples: Sequence[SampleMetrics],
    output_dir: Path,
    bins: int = 30,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    data = {
        "x_px": {"flow": [], "mgn": []},
        "y_py": {"flow": [], "mgn": []},
        "z_pz": {"flow": [], "mgn": []},
    }

    for s in samples:
        for slice_name in ["x_px", "y_py", "z_pz"]:
            data[slice_name]["flow"].append(s.density_mse_flow[slice_name])
            data[slice_name]["mgn"].append(s.density_mse_mgn[slice_name])

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    slice_names = ["x_px", "y_py", "z_pz"]
    colors = {"flow": "firebrick", "mgn": "forestgreen"}

    for ax, slice_name in zip(axes, slice_names):
        for model in ["flow", "mgn"]:
            vals = data[slice_name][model]
            ax.hist(
                vals,
                bins=bins,
                alpha=0.5,
                label=model.upper(),
                density=True,
                color=colors[model],
            )
            mean_val = np.mean(vals)
            ax.axvline(mean_val, color=colors[model], linestyle="--", linewidth=1)

        ax.set_title(f"MSE Distribution: {slice_name}")
        ax.set_xlabel("MSE")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Phase Space Density MSE Distributions per Slice")
    fig.tight_layout()
    fig.savefig(output_dir / "density_mse_distributions_per_slice.png", dpi=200)
    plt.close(fig)

    print("\nPhase Space Density MSE Summary (per slice):")
    for slice_name in slice_names:
        print(f"  Slice {slice_name}:")
        for model in ["flow", "mgn"]:
            vals = data[slice_name][model]
            print(f"    {model.upper()} mean MSE: {np.mean(vals):.4e}")


def plot_2d_density_comparison(
    samples: Sequence[SampleMetrics],
    output_dir: Path,
    bins: int = 50,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = {
        "x_px": (0, 3),
        "y_py": (1, 4),
        "z_pz": (2, 5),
    }

    for idx, sample in enumerate(samples, start=1):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        for row_idx, (slice_name, (dim1, dim2)) in enumerate(pairs.items()):
            t1, t2 = sample.true_final[:, dim1], sample.true_final[:, dim2]
            f1, f2 = sample.flow_pred[:, dim1], sample.flow_pred[:, dim2]
            m1, m2 = sample.mgn_pred[:, dim1], sample.mgn_pred[:, dim2]

            all_1 = np.concatenate([t1, f1, m1])
            all_2 = np.concatenate([t2, f2, m2])
            min1, max1 = all_1.min(), all_1.max()
            min2, max2 = all_2.min(), all_2.max()
            range_limits = [[min1, max1], [min2, max2]]

            # True
            ax_true = axes[row_idx, 0]
            ax_true.hist2d(
                t1,
                t2,
                bins=bins,
                range=range_limits,
                density=True,
                cmap="viridis",
            )
            ax_true.set_title(f"True {slice_name}")

            # Flow
            ax_flow = axes[row_idx, 1]
            ax_flow.hist2d(
                f1,
                f2,
                bins=bins,
                range=range_limits,
                density=True,
                cmap="viridis",
            )
            ax_flow.set_title(
                f"Flow {slice_name}\nMSE: {sample.density_mse_flow[slice_name]:.2e}"
            )

            # MGN
            ax_mgn = axes[row_idx, 2]
            ax_mgn.hist2d(
                m1,
                m2,
                bins=bins,
                range=range_limits,
                density=True,
                cmap="viridis",
            )
            ax_mgn.set_title(
                f"MGN {slice_name}\nMSE: {sample.density_mse_mgn[slice_name]:.2e}"
            )

        fig.suptitle(f"2D Phase Space Density – Sample {idx}: {sample.file_path.name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{idx:02d}_2d_density.png", dpi=200)
        plt.close(fig)


def save_per_sample_beam_metrics(
    samples: Sequence[SampleMetrics],
    output_path: Path,
    baseline_abs: np.ndarray,
) -> None:
    records = []
    for sample in samples:
        flow_diff = sample.beam_flow - sample.beam_true
        mgn_diff = sample.beam_mgn - sample.beam_true
        for i in range(6):
            for j in range(6):
                denom = max(float(baseline_abs[i, j]), BEAM_EPS)
                records.append(
                    {
                        "file": sample.file_path.name,
                        "i": i,
                        "j": j,
                        "model": "flow",
                        "abs_error": float(abs(flow_diff[i, j])),
                        "signed_error": float(flow_diff[i, j]),
                        "abs_percent_error": float(
                            abs(flow_diff[i, j]) / denom * 100.0
                        ),
                        "signed_percent_error": float(flow_diff[i, j] / denom * 100.0),
                    }
                )
                records.append(
                    {
                        "file": sample.file_path.name,
                        "i": i,
                        "j": j,
                        "model": "mgn",
                        "abs_error": float(abs(mgn_diff[i, j])),
                        "signed_error": float(mgn_diff[i, j]),
                        "abs_percent_error": float(abs(mgn_diff[i, j]) / denom * 100.0),
                        "signed_percent_error": float(mgn_diff[i, j] / denom * 100.0),
                    }
                )
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_path, index=False)


def save_per_sample_emittance_metrics(
    samples: Sequence[SampleMetrics],
    output_path: Path,
    metrics_df: Optional[pd.DataFrame] = None,
) -> None:
    df = metrics_df if metrics_df is not None else collect_emittance_metrics(samples)
    df.to_csv(output_path, index=False)


def save_density_mse_summary(
    samples: Sequence[SampleMetrics],
    output_path: Path,
) -> None:
    records = []
    slice_names = ["x_px", "y_py", "z_pz"]

    # Collect data
    data = {
        "x_px": {"flow": [], "mgn": []},
        "y_py": {"flow": [], "mgn": []},
        "z_pz": {"flow": [], "mgn": []},
    }

    for s in samples:
        for slice_name in slice_names:
            data[slice_name]["flow"].append(s.density_mse_flow[slice_name])
            data[slice_name]["mgn"].append(s.density_mse_mgn[slice_name])

    for slice_name in slice_names:
        for model in ["flow", "mgn"]:
            vals = np.array(data[slice_name][model])
            if vals.size == 0:
                continue

            records.append(
                {
                    "slice": slice_name,
                    "model": model,
                    "mean_mse": float(np.mean(vals)),
                    "median_mse": float(np.median(vals)),
                    "max_mse": float(np.max(vals)),
                    "min_mse": float(np.min(vals)),
                    "range_mse": float(np.max(vals) - np.min(vals)),
                    "std_mse": float(np.std(vals)),
                }
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    particle_dir = args.particle_dir.expanduser().resolve()
    if not particle_dir.exists():
        raise FileNotFoundError(f"Particle directory not found: {particle_dir}")

    flow_checkpoint = args.flow_checkpoint.expanduser().resolve()
    flow_scalers = args.flow_scalers.expanduser().resolve()
    mgn_checkpoint = args.mgn_checkpoint.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    settings_path = (
        args.settings.expanduser().resolve() if args.settings is not None else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        args.subsample_seed = args.seed
        args.flow_seed = args.seed

    print(
        "Using seeds -> subsample/MGN: "
        f"{args.subsample_seed}, flow: {args.flow_seed}"
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    flow_model, scaler_final, scaler_condition = load_flow_model(
        flow_checkpoint, flow_scalers, device
    )
    print("Loaded conditional flow model")

    metadata_bundle = maybe_load_metadata(metadata_path)
    if metadata_bundle is None:
        raise ValueError("Metadata is required for MeshGraphNet evaluation")
    norm_stats, edge_config = prepare_edge_config(metadata_bundle, args.k)

    settings_dim = args.settings_dim
    if settings_dim is None:
        settings_dim = infer_settings_dim(settings_path)
    settings_tensor = load_settings(settings_path, settings_dim).float()

    edge_attr_method = args.edge_attr_method
    if edge_attr_method is None:
        hyper = parse_hyperparams_from_checkpoint(mgn_checkpoint)
        edge_attr_method = hyper.get("edge_attr_method")

    particle_files = list_particle_files(particle_dir)
    if args.limit is not None:
        particle_files = particle_files[: args.limit]
    if not particle_files:
        raise ValueError(f"No particle files found in {particle_dir}")

    print(f"Evaluating {len(particle_files)} files")

    representative: List[SampleMetrics] = []
    results: List[SampleMetrics] = []

    mgn_model: Optional[MeshGraphNet] = None

    for idx, file_path in enumerate(particle_files):
        print(f"[{idx + 1}/{len(particle_files)}] Processing {file_path.name}")

        trajectory = load_trajectory(file_path)

        initial_full = trajectory[0].cpu().numpy()
        final_full = trajectory[-1].cpu().numpy()

        try:
            initial_np, final_np, valid_idx = sanitise_particle_pair(
                initial_full, final_full, file_path
            )
        except ValueError as exc:
            print(f"  Skipping {file_path.name}: {exc}")
            continue

        initial_sub, local_indices = subsample_particles(
            initial_np, args.n_particles, args.subsample_seed
        )
        final_sub = final_np[local_indices]
        indices = valid_idx[local_indices]

        initial_raw = torch.from_numpy(initial_sub.astype(np.float32))
        if norm_stats is not None:
            initial_norm = normalise_tensor(initial_raw, norm_stats["initial"])
        else:
            initial_norm = initial_raw.clone()

        graph = build_graph(
            initial_norm, settings_tensor, dict(edge_config), edge_attr_method
        )

        if mgn_model is None:
            mgn_model = initialise_mgn_model(
                graph=graph,
                checkpoint_path=mgn_checkpoint,
                hidden_dim_override=args.hidden_dim,
                num_layers_override=args.num_layers,
                device=device,
            )
            print("Loaded MeshGraphNet model")
        else:
            expected_node = getattr(mgn_model, "_node_in_dim")
            if expected_node != graph.x.shape[1]:
                raise ValueError("Node feature dimension mismatch across samples")

        try:
            sample_metrics = evaluate_sample(
                file_path=file_path,
                initial_sub=initial_sub,
                final_sub=final_sub,
                indices=indices,
                flow_seed=args.flow_seed,
                flow_model=flow_model,
                scaler_final=scaler_final,
                scaler_condition=scaler_condition,
                mgn_model=mgn_model,
                settings=settings_tensor,
                norm_stats=norm_stats,
                edge_config=dict(edge_config),
                edge_attr_method=edge_attr_method,
                device=device,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Error evaluating {file_path.name}: {exc}")
            continue

        results.append(sample_metrics)
        if len(representative) < args.representative_count:
            representative.append(sample_metrics)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":  # type: ignore[attr-defined]
            torch.mps.empty_cache()
        gc.collect()

    if not results:
        raise RuntimeError("No samples were successfully evaluated")

    beam_true, beam_flow, beam_mgn = stack_beam_matrices(results)
    summary_df, baseline_abs = beam_summary(beam_true, beam_flow, beam_mgn)
    summary_path = output_dir / "beam_matrix_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved beam matrix summary to {summary_path}")

    emittance_df = emittance_summary(results)
    emittance_path = output_dir / "emittance_summary.csv"
    emittance_df.to_csv(emittance_path, index=False)
    print(f"Saved emittance summary to {emittance_path}")

    plot_beam_heatmaps(
        summary_df,
        output_dir,
        value_col="median_abs_error",
        filename_prefix="beam_median_error_heatmap",
        title_prefix="Median |Beam Error|",
        colorbar_label="Absolute error",
        formatter=lambda v: f"{v:.2e}",
    )
    plot_beam_heatmaps(
        summary_df,
        output_dir,
        value_col="median_abs_percent_error",
        filename_prefix="beam_median_percent_error_heatmap",
        title_prefix="Median |Beam Error| (%)",
        colorbar_label="Percent error [%]",
        formatter=lambda v: f"{v:.1f}%",
    )
    plot_beam_line(
        summary_df,
        output_dir / "beam_median_error_by_entry.png",
        value_col="median_abs_error",
        ylabel="Median |error|",
        title="Beam matrix entry median absolute error",
    )
    plot_beam_line(
        summary_df,
        output_dir / "beam_median_percent_error_by_entry.png",
        value_col="median_abs_percent_error",
        ylabel="Median |error| [%]",
        title="Beam matrix entry median absolute percent error",
        as_percent=True,
    )

    emittance_metrics_df = collect_emittance_metrics(results)
    plot_emittance_scatter(results, output_dir / "emittance_plots")
    plot_emittance_error_histograms(
        emittance_metrics_df, output_dir / "emittance_plots"
    )

    plot_density_mse_distributions(results, output_dir / "density_mse")
    save_density_mse_summary(results, output_dir / "density_mse_summary.csv")

    plot_representative_samples(representative, output_dir / "representative_samples")
    plot_2d_density_comparison(representative, output_dir / "representative_samples_2d_density")
    
    # Plot highest emittance error samples for both models combined
    top_sixd_samples = select_top_emittance_samples(results, count=5, model="both")
    if top_sixd_samples:
        plot_representative_samples(
            top_sixd_samples,
            output_dir / "representative_samples_high_6d",
        )
    
    # Plot highest emittance error samples for Flow model
    top_flow_samples = select_top_emittance_samples(results, count=5, model="flow")
    if top_flow_samples:
        plot_representative_samples(
            top_flow_samples,
            output_dir / "representative_samples_high_flow",
        )
    
    # Plot highest emittance error samples for MGN model
    top_mgn_samples = select_top_emittance_samples(results, count=5, model="mgn")
    if top_mgn_samples:
        plot_representative_samples(
            top_mgn_samples,
            output_dir / "representative_samples_high_mgn",
        )
    
    plot_beam_entry_distributions(representative, output_dir / "representative_samples")

    if args.save_per_sample:
        save_per_sample_beam_metrics(
            results, output_dir / "per_sample_beam_metrics.csv", baseline_abs
        )
        save_per_sample_emittance_metrics(
            results,
            output_dir / "per_sample_emittance_metrics.csv",
            metrics_df=emittance_metrics_df,
        )

    print("Evaluation complete")


if __name__ == "__main__":
    main()
