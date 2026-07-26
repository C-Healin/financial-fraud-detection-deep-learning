"""PyTorch MLP with learned embeddings for categorical transaction features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..constants import TARGET
from ..metrics import classification_metrics, select_threshold
from ..preprocessing import FraudPreprocessor


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        n_numerical: int,
        embedding_dim: int = 16,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities]
        )
        in_features = len(cardinalities) * embedding_dim + n_numerical
        layers: list[nn.Module] = []
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden),
                    nn.PReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        embedded = [layer(categorical[:, idx]) for idx, layer in enumerate(self.embeddings)]
        x = torch.cat(embedded + [numerical], dim=1)
        return self.network(x).squeeze(1)


def _loader(
    preprocessor: FraudPreprocessor,
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    cats, nums = preprocessor.transform(df)
    target = df[TARGET].to_numpy(dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(cats).long(),
        torch.from_numpy(nums).float(),
        torch.from_numpy(target).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for cats, nums, _ in loader:
            logits = model(cats.to(device), nums.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def train_mlp_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifact_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 512,
) -> tuple[dict[str, object], np.ndarray, FraudPreprocessor, EmbeddingMLP]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = FraudPreprocessor.create().fit(train_df)
    train_loader = _loader(preprocessor, train_df, batch_size, True)
    val_loader = _loader(preprocessor, val_df, batch_size, False)
    test_loader = _loader(preprocessor, test_df, batch_size, False)

    model = EmbeddingMLP(
        cardinalities=preprocessor.categorical_cardinalities,
        n_numerical=len(preprocessor.numerical_features),
    ).to(device)

    y_train = train_df[TARGET].to_numpy(dtype=int)
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    best_state = None
    best_ap = -np.inf
    patience = 3
    stale = 0
    from sklearn.metrics import average_precision_score

    for _epoch in range(epochs):
        model.train()
        for cats, nums, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(cats.to(device), nums.to(device))
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            optimizer.step()

        val_probability = _predict(model, val_loader, device)
        val_ap = average_precision_score(val_df[TARGET].to_numpy(dtype=int), val_probability)
        if val_ap > best_ap + 1e-5:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    val_probability = _predict(model, val_loader, device)
    threshold = select_threshold(val_df[TARGET].to_numpy(dtype=int), val_probability)
    test_probability = _predict(model, test_loader, device)
    metrics = classification_metrics(test_df[TARGET].to_numpy(dtype=int), test_probability, threshold)
    metrics["model"] = "Embedding MLP"

    preprocessor.save(artifact_dir / "mlp_preprocessor.joblib")
    torch.save(model.state_dict(), artifact_dir / "mlp_model.pt")
    (artifact_dir / "mlp_metadata.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "cardinalities": preprocessor.categorical_cardinalities,
                "n_numerical": len(preprocessor.numerical_features),
                "embedding_dim": 16,
                "hidden_dims": [256, 128, 64],
                "dropout": 0.20,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics, test_probability, preprocessor, model
