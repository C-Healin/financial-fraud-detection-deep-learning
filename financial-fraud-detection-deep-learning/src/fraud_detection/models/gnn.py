"""A dependency-light bipartite GraphSAGE edge-classification model in PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..constants import CARD_COLUMN, MERCHANT_COLUMN, NUMERICAL_FEATURES, TARGET
from ..metrics import classification_metrics, select_threshold


class SageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_mean = torch.sparse.mm(adjacency, x)
        return torch.relu(self.self_linear(x) + self.neighbor_linear(neighbor_mean))


class BipartiteFraudGNN(nn.Module):
    def __init__(self, n_nodes: int, edge_feature_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(n_nodes, hidden_dim)
        self.sage1 = SageLayer(hidden_dim, hidden_dim)
        self.sage2 = SageLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.15)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_feature_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.node_embedding.weight
        x = self.dropout(self.sage1(x, adjacency))
        return self.sage2(x, adjacency)

    def score_edges(
        self,
        node_embeddings: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        h_src = node_embeddings[source]
        h_dst = node_embeddings[destination]
        pair = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst, edge_features], dim=1)
        return self.edge_classifier(pair).squeeze(1)


def _node_maps(frames: list[pd.DataFrame]) -> tuple[dict[str, int], dict[str, int]]:
    cards = sorted(set().union(*(set(frame[CARD_COLUMN].astype(str)) for frame in frames)))
    merchants = sorted(set().union(*(set(frame[MERCHANT_COLUMN].astype(str)) for frame in frames)))
    card_map = {value: idx for idx, value in enumerate(cards)}
    offset = len(card_map)
    merchant_map = {value: offset + idx for idx, value in enumerate(merchants)}
    return card_map, merchant_map


def _edge_indices(
    df: pd.DataFrame,
    card_map: dict[str, int],
    merchant_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    src = df[CARD_COLUMN].astype(str).map(card_map).to_numpy(dtype=np.int64)
    dst = df[MERCHANT_COLUMN].astype(str).map(merchant_map).to_numpy(dtype=np.int64)
    return src, dst


def _normalized_adjacency(
    n_nodes: int,
    source: np.ndarray,
    destination: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    row = np.concatenate([source, destination, np.arange(n_nodes)])
    col = np.concatenate([destination, source, np.arange(n_nodes)])
    degree = np.bincount(row, minlength=n_nodes).astype(np.float32)
    values = 1.0 / np.maximum(degree[row], 1.0)
    indices = torch.tensor(np.vstack([row, col]), dtype=torch.long, device=device)
    vals = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, vals, (n_nodes, n_nodes), device=device).coalesce()


def train_gnn_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifact_dir: str | Path,
    epochs: int = 24,
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    card_map, merchant_map = _node_maps([train_df, val_df, test_df])
    train_src, train_dst = _edge_indices(train_df, card_map, merchant_map)
    val_src, val_dst = _edge_indices(val_df, card_map, merchant_map)
    test_src, test_dst = _edge_indices(test_df, card_map, merchant_map)
    n_nodes = len(card_map) + len(merchant_map)
    adjacency = _normalized_adjacency(n_nodes, train_src, train_dst, device)

    scaler = StandardScaler().fit(train_df[NUMERICAL_FEATURES].astype(float))
    train_edge = torch.tensor(scaler.transform(train_df[NUMERICAL_FEATURES]), dtype=torch.float32, device=device)
    val_edge = torch.tensor(scaler.transform(val_df[NUMERICAL_FEATURES]), dtype=torch.float32, device=device)
    test_edge = torch.tensor(scaler.transform(test_df[NUMERICAL_FEATURES]), dtype=torch.float32, device=device)
    y_train = torch.tensor(train_df[TARGET].to_numpy(dtype=np.float32), device=device)
    y_val_np = val_df[TARGET].to_numpy(dtype=int)
    y_test_np = test_df[TARGET].to_numpy(dtype=int)

    train_src_t = torch.tensor(train_src, dtype=torch.long, device=device)
    train_dst_t = torch.tensor(train_dst, dtype=torch.long, device=device)
    val_src_t = torch.tensor(val_src, dtype=torch.long, device=device)
    val_dst_t = torch.tensor(val_dst, dtype=torch.long, device=device)
    test_src_t = torch.tensor(test_src, dtype=torch.long, device=device)
    test_dst_t = torch.tensor(test_dst, dtype=torch.long, device=device)

    model = BipartiteFraudGNN(n_nodes=n_nodes, edge_feature_dim=len(NUMERICAL_FEATURES)).to(device)
    positives = max(int(train_df[TARGET].sum()), 1)
    negatives = max(len(train_df) - positives, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    from sklearn.metrics import average_precision_score
    best_state = None
    best_ap = -np.inf
    patience = 5
    stale = 0
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.encode(adjacency)
        logits = model.score_edges(embeddings, train_src_t, train_dst_t, train_edge)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            embeddings = model.encode(adjacency)
            val_prob = torch.sigmoid(
                model.score_edges(embeddings, val_src_t, val_dst_t, val_edge)
            ).cpu().numpy()
        ap = average_precision_score(y_val_np, val_prob)
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.inference_mode():
        embeddings = model.encode(adjacency)
        val_probability = torch.sigmoid(
            model.score_edges(embeddings, val_src_t, val_dst_t, val_edge)
        ).cpu().numpy()
        test_probability = torch.sigmoid(
            model.score_edges(embeddings, test_src_t, test_dst_t, test_edge)
        ).cpu().numpy()
        node_embeddings = embeddings.cpu().numpy()

    threshold = select_threshold(y_val_np, val_probability)
    metrics = classification_metrics(y_test_np, test_probability, threshold)
    metrics["model"] = "GraphSAGE"

    torch.save(model.state_dict(), artifact_dir / "gnn_model.pt")
    np.save(artifact_dir / "gnn_node_embeddings.npy", node_embeddings)
    import joblib
    joblib.dump(scaler, artifact_dir / "gnn_edge_scaler.joblib")
    metadata = {
        "threshold": threshold,
        "n_nodes": n_nodes,
        "hidden_dim": 48,
        "edge_feature_dim": len(NUMERICAL_FEATURES),
        "card_map": card_map,
        "merchant_map": merchant_map,
        "numerical_features": NUMERICAL_FEATURES,
    }
    (artifact_dir / "gnn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics, test_probability, {
        "embeddings": node_embeddings,
        "card_map": card_map,
        "merchant_map": merchant_map,
        "scaler": scaler,
    }
