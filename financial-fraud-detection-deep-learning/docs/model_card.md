# Model Card

## Models

1. **XGBoost baseline** — strong tabular benchmark with class weighting.
2. **Embedding MLP** — PyTorch network with trainable embeddings for categorical features.
3. **GraphSAGE** — card–merchant bipartite graph with edge-level fraud classification.
4. **GNN embeddings + XGBoost** — graph representations concatenated with tabular features.

## Primary metric

PR-AUC / Average Precision is the primary selection metric because fraud is a rare-event classification problem. ROC-AUC, precision, recall, F1, and confusion matrices are reported as secondary metrics.

## Threshold policy

The classification threshold is chosen on validation data by maximizing F1. A real financial institution should replace this with a cost-sensitive policy based on fraud loss, review capacity, customer friction, and regulatory requirements.

## Limitations

- Synthetic data does not represent a real fraud distribution.
- The graph model is transductive for known card and merchant nodes.
- No fairness, drift, adversarial robustness, or privacy audit is included.
- The demo is not a credit decision system and should not be used for real financial decisions.
