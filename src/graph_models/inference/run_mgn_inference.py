#!/usr/bin/env python
"""
Run MeshGraphNet inference on a raw particle trajectory file.

Given a checkpoint produced by src/graph_models/train.py and a particle
trajectory (the same format consumed by train_norm_flow_conditional.py), this
script constructs a single PyG graph on-the-fly, performs forward inference,
optionally denormalises the prediction, and saves tensor outputs that match the
train_norm_flow workflow (initial particles, final particles, predicted final
particles, and conditioning statistics).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.graph_models.dataloaders import GraphDataset
from src.graph_models.models.graph_networks import MeshGraphNet
from src.preprocessing.generate_graphs_from_sequence_particles import (
    _build_edges_by_distance,
    _build_edges_knn,
    settings_dict_to_tensor,
)

# Torch >= 2.6 defaults weights_only=True. We need full objects.
TORCH_LOAD_OPTS = {"weights_only": False}
EPS = 1e-10


def load_trajectory(file_path: Path) -> torch.Tensor:
    """Load a trajectory .pt file in the same fashion as the flow trainings."""
    trajectory = torch.load(file_path, **TORCH_LOAD_OPTS)
    if isinstance(trajectory, dict):
        if "trajectory" in trajectory:
            trajectory = trajectory["trajectory"]
        elif "particles" in trajectory:
            trajectory = trajectory["particles"]
        elif "data" in trajectory:
            trajectory = trajectory["data"]
    if not isinstance(trajectory, torch.Tensor):
        raise TypeError(f"Unsupported trajectory container: {type(trajectory)}")
    if trajectory.ndim != 3:
        raise ValueError(f"Expected [T, N, 6] tensor, got shape {trajectory.shape}")
    return trajectory


def subsample_particles(
    particles: np.ndarray,
    n_target: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically subsample N particles, returning the subset and indices."""
    rng = np.random.default_rng(seed)
    n_particles = particles.shape[0]
    if n_particles <= n_target:
        indices = np.arange(n_particles)
    else:
        indices = np.sort(rng.choice(n_particles, n_target, replace=False))
    return particles[indices], indices


def compute_condition_features(phase_space: np.ndarray) -> np.ndarray:
    """Compute the 45 conditioning statistics used by the flow workflow."""
    mean = np.mean(phase_space, axis=0)
    std = np.std(phase_space, axis=0)
    cov = np.cov(phase_space.T)
    cov_triu = cov[np.triu_indices(6)]
    skew = scipy_skew(phase_space, axis=0)
    kurt = scipy_kurtosis(phase_space, axis=0)
    return np.concatenate([mean, std, cov_triu, skew, kurt])


def scipy_skew(data: np.ndarray, axis: int) -> np.ndarray:
    from scipy.stats import skew
    return skew(data, axis=axis)


def scipy_kurtosis(data: np.ndarray, axis: int) -> np.ndarray:
    from scipy.stats import kurtosis
    return kurtosis(data, axis=axis)


def load_settings(settings_path: Optional[Path], expected_dim: int) -> torch.Tensor:
    """Load accelerator settings vector; default to zeros if not provided."""
    if settings_path is None:
        return torch.zeros(expected_dim)
    payload = torch.load(settings_path, **TORCH_LOAD_OPTS)
    if isinstance(payload, dict):
        tensor = settings_dict_to_tensor(payload).float()
    else:
        tensor = torch.as_tensor(payload, dtype=torch.float32)
    if tensor.numel() != expected_dim:
        raise ValueError(f"Expected settings dim {expected_dim}, found {tensor.numel()}")
    return tensor.view(-1)


class _EdgeAttrHelper:
    """Lightweight adapter to reuse GraphDataset edge-attribute logic."""

    def __init__(self, method: str) -> None:
        self.edge_attr_method = method


def build_graph(
    initial_particles: torch.Tensor,
    settings: torch.Tensor,
    edge_config: Dict[str, object],
    edge_attr_method: Optional[str],
) -> Data:
    """Create a PyG Data graph using the same utilities as dataset preprocessing."""
    node_features = initial_particles.float()
    num_nodes = node_features.shape[0]

    # Append settings to each node feature, mirroring GraphDataset.__getitem__.
    settings_expanded = settings.view(1, -1).repeat(num_nodes, 1)
    x = torch.cat([node_features, settings_expanded], dim=1)

    pos = node_features[:, :3]

    # Build edges using the same helper employed during preprocessing.
    edge_method = edge_config.get("edge_method", "knn")
    weighted_edge = bool(edge_config.get("weighted_edge", False))
    if edge_method == "dist":
        distance_threshold = float(edge_config.get("distance_threshold", float("inf")))
        edge_index, edge_weight = _build_edges_by_distance(
            node_features.t(), distance_threshold, weighted_edge
        )
    else:
        k = int(edge_config.get("k", 5))
        edge_index, edge_weight = _build_edges_knn(node_features.t(), k, weighted_edge)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        pos=pos,
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )

    if edge_attr_method:
        helper = _EdgeAttrHelper(edge_attr_method)
        GraphDataset._compute_edge_attr(helper, data, idx=-1)
    else:
        data.edge_attr = None

    return data


def parse_hyperparams_from_checkpoint(checkpoint_path: Path) -> Dict[str, str]:
    folder = checkpoint_path.parent.parent.name
    parts = folder.split("_")

    def extract(prefix: str) -> Optional[str]:
        for part in parts:
            if part.startswith(prefix):
                return part[len(prefix):]
        return None

    return {
        "hidden_dim": extract("h"),
        "num_layers": extract("ly"),
        "edge_attr_method": (extract("ea") or "v1"),
    }


def maybe_load_metadata(
    metadata_path: Optional[Path],
) -> Optional[Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
    if metadata_path is None:
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    global_mean = torch.tensor(meta["global_mean"], dtype=torch.float32)
    global_std = torch.tensor(meta["global_std"], dtype=torch.float32)

    if global_mean.ndim == 1:
        mean_initial = mean_final = global_mean
        std_initial = std_final = global_std
    else:
        mean_initial = global_mean[0]
        std_initial = global_std[0]
        mean_final = global_mean[-1]
        std_final = global_std[-1]

    norm_stats = {
        "initial": (mean_initial, std_initial),
        "final": (mean_final, std_final),
    }

    edge_config = {
        "edge_method": meta.get("edge_method", "knn"),
        "weighted_edge": meta.get("weighted_edge", False),
        "k": meta.get("k", 5),
        "distance_threshold": meta.get("distance_threshold", float("inf")),
    }

    return {"stats": norm_stats, "edge_config": edge_config}


def normalise_tensor(data: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, std = stats
    mean = mean.to(data.device, dtype=data.dtype)
    std = std.to(data.device, dtype=data.dtype)
    return (data - mean) / (std + EPS)


def denormalise_tensor(data: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, std = stats
    mean = mean.to(data.device, dtype=data.dtype)
    std = std.to(data.device, dtype=data.dtype)
    return data * std + mean


def denormalise(pred: torch.Tensor, stats: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if stats is None:
        return pred
    return denormalise_tensor(pred, stats)


def run_inference(args: argparse.Namespace) -> Path:
    trajectory = load_trajectory(args.particle_data)
    initial_np = trajectory[0].cpu().numpy()
    final_np = trajectory[-1].cpu().numpy()

    initial_sub, indices = subsample_particles(initial_np, args.n_particles, args.seed)
    if final_np.shape[0] < len(indices):
        raise ValueError(
            "Final particle set is smaller than the requested subsample; ensure trajectories have matching counts."
        )
    final_sub = final_np[indices]

    initial_raw = torch.from_numpy(initial_sub.astype(np.float32))
    settings = load_settings(args.settings, args.settings_dim)

    hyper = parse_hyperparams_from_checkpoint(args.checkpoint.expanduser().resolve())
    hidden_dim = int(hyper["hidden_dim"]) if hyper["hidden_dim"] is not None else args.hidden_dim
    num_layers = int(hyper["num_layers"]) if hyper["num_layers"] is not None else args.num_layers
    edge_method = hyper["edge_attr_method"] if hyper["edge_attr_method"] is not None else args.edge_attr_method

    metadata_bundle = maybe_load_metadata(args.metadata)

    if metadata_bundle is not None:
        norm_stats = metadata_bundle["stats"]
        edge_config = metadata_bundle["edge_config"]
    else:
        norm_stats = None
        edge_config = {
            "edge_method": "knn",
            "weighted_edge": False,
            "k": args.k,
            "distance_threshold": float("inf"),
        }

    final_raw = torch.from_numpy(final_sub.astype(np.float32))

    if norm_stats is not None:
        initial_norm = normalise_tensor(initial_raw, norm_stats["initial"])
        final_norm = normalise_tensor(final_raw, norm_stats["final"])
    else:
        initial_norm = initial_raw.clone()
        final_norm = final_raw.clone()

    # Ensure edge construction configuration honors CLI overrides when provided.
    if args.k is not None:
        edge_config["k"] = args.k
    graph = build_graph(initial_norm, settings, edge_config, edge_method)

    node_in_dim = graph.x.shape[1]
    edge_in_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else 0

    model = MeshGraphNet(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        node_out_dim=6,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", **TORCH_LOAD_OPTS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    graph = graph.to(args.device)
    with torch.no_grad():
        pred_norm = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

    pred_norm = pred_norm.cpu()

    final_stats = norm_stats["final"] if norm_stats is not None else None
    initial_stats = norm_stats["initial"] if norm_stats is not None else None

    pred_denorm = denormalise(pred_norm, final_stats)
    final_denorm = denormalise_tensor(final_norm, final_stats) if final_stats is not None else final_norm

    condition = compute_condition_features(initial_sub)

    outputs = {
        "initial_particles": initial_sub.astype(np.float32),
        "initial_particles_normalized": initial_norm.numpy().astype(np.float32),
        "pred_final": pred_denorm.numpy().astype(np.float32),
        "pred_final_normalized": pred_norm.numpy().astype(np.float32),
        "true_final": final_denorm.numpy().astype(np.float32) if final_denorm is not None else None,
        "true_final_normalized": final_norm.numpy().astype(np.float32),
        "condition_stats": condition.astype(np.float32),
        "settings": settings.numpy().astype(np.float32),
        "indices": indices.astype(np.int64),
    }

    npz_payload = {k: v for k, v in outputs.items() if v is not None}
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **npz_payload)

    return output_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MeshGraphNet inference on particle data")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--particle-data", type=Path, required=True, help="Path to particle trajectory .pt file")
    parser.add_argument("--settings", type=Path, default=None, help="Optional accelerator settings tensor (.pt)")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata_final.json for denormalisation")
    parser.add_argument("--output", type=Path, default=Path("mgn_inference_output.npz"), help="Output .npz path")
    parser.add_argument("--n-particles", type=int, default=2000, help="Particles to subsample")
    parser.add_argument("--seed", type=int, default=63, help="Subsampling RNG seed")
    parser.add_argument("--k", type=int, default=5, help="k-NN graph size")
    parser.add_argument("--settings-dim", type=int, default=6, help="Expected settings dimension")
    parser.add_argument("--edge-attr-method", type=str, default="v1", help="Fallback edge_attr method")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Fallback hidden dim if not encoded in checkpoint")
    parser.add_argument("--num-layers", type=int, default=6, help="Fallback num layers if not encoded in checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.particle_data = args.particle_data.expanduser().resolve()
    args.settings = args.settings.expanduser().resolve() if args.settings is not None else None
    args.metadata = args.metadata.expanduser().resolve() if args.metadata is not None else None
    args.output = args.output.expanduser().resolve()

    output_path = run_inference(args)
    print(f"Saved inference outputs to {output_path}")


if __name__ == "__main__":
    main()
