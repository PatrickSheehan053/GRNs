"""
HYPHAE / 2_run_hyphae_worker.py
────────────────────────────────
GPU-based GRN inference engine using perturbation-aware attention.

Trains a neural network to learn GRN edge weights by optimizing
for perturbation prediction accuracy via differentiable graph
propagation. Each SLURM array task trains on a different cell
subsample for bootstrap stability estimation.

Requires perturbation labels in adata.obs — will exit with a
clear error if none are found, since the training signal comes
entirely from perturbation data.
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Perturbation label detection (shared with GuanLab Experimental) ────────────────────

KNOWN_PERT_COLUMNS = [
    "perturbation", "condition", "gene", "guide_identity",
    "perturbation_gene", "target_gene", "pert", "intervention",
    "grna_target", "gRNA_target", "pertTarget",
]
KNOWN_CONTROL_VALUES = [
    "control", "ctrl", "non-targeting", "non_targeting",
    "nt", "NT", "negative_control", "unperturbed",
    "non-essential", "safe-targeting", "safe_targeting",
]


def detect_perturbation_column(adata):
    for col in KNOWN_PERT_COLUMNS:
        if col in adata.obs.columns:
            return col
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object or adata.obs[col].dtype.name == "category":
            values = set(adata.obs[col].astype(str).str.lower().unique())
            if values.intersection(set(v.lower() for v in KNOWN_CONTROL_VALUES)):
                return col
    return None


def identify_control_mask(adata, pert_col):
    labels_lower = adata.obs[pert_col].astype(str).str.lower()
    return labels_lower.isin([v.lower() for v in KNOWN_CONTROL_VALUES]).values


# ── Stage 1: Perturbation Signature Computation ─────────────────────────────

def compute_perturbation_signatures(X, gene_names, pert_col_values,
                                    control_mask, min_cells=5,
                                    efficiency_threshold=0.5):
    """
    Compute the perturbation signature matrix S and gene-level features.

    S[p, j] = mean(gene_j in perturbed_p cells) - mean(gene_j in control cells)

    Returns:
        S: (n_pert, n_genes) perturbation signature matrix
        pert_gene_indices: list of gene indices for each perturbation
        pert_names: list of perturbation gene names
        gene_features: (n_genes, n_features) feature matrix for embeddings
        pert_efficiency: dict of gene_name -> knockdown efficiency
    """
    gene_set = set(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    # Identify control cells
    ctrl_idx = np.where(control_mask)[0]
    ctrl_mean = np.mean(X[ctrl_idx], axis=0)  # (n_genes,)
    ctrl_std = np.std(X[ctrl_idx], axis=0) + 1e-8

    # Build per-perturbation cell groups
    pert_groups = {}
    for cell_idx in range(len(pert_col_values)):
        if control_mask[cell_idx]:
            continue
        label = str(pert_col_values[cell_idx])
        if label in gene_set:
            if label not in pert_groups:
                pert_groups[label] = []
            pert_groups[label].append(cell_idx)

    # Filter by cell count and compute efficiency
    pert_names = []
    pert_gene_indices = []
    pert_efficiency = {}
    signatures = []

    for gene_name, cell_indices in sorted(pert_groups.items()):
        if len(cell_indices) < min_cells:
            continue
        if gene_name not in gene_to_idx:
            continue

        gene_idx = gene_to_idx[gene_name]
        cell_arr = np.array(cell_indices)

        # Knockdown efficiency: how many SDs below control mean
        pert_expr_self = np.mean(X[cell_arr, gene_idx])
        efficiency = (ctrl_mean[gene_idx] - pert_expr_self) / ctrl_std[gene_idx]
        pert_efficiency[gene_name] = float(efficiency)

        if efficiency < efficiency_threshold:
            continue

        # Perturbation signature: expression change for all genes
        pert_mean = np.mean(X[cell_arr], axis=0)
        sig = pert_mean - ctrl_mean
        signatures.append(sig)
        pert_names.append(gene_name)
        pert_gene_indices.append(gene_idx)

    S = np.stack(signatures, axis=0)  # (n_pert, n_genes)

    # Compute gene-level features for embedding initialization
    # Feature 1: mean expression in controls (normalized)
    feat_mean = ctrl_mean / (np.max(np.abs(ctrl_mean)) + 1e-8)
    # Feature 2: expression variance in controls (normalized)
    feat_var = ctrl_std / (np.max(ctrl_std) + 1e-8)
    # Feature 3: mean absolute perturbation response
    feat_response = np.mean(np.abs(S), axis=0)
    feat_response = feat_response / (np.max(feat_response) + 1e-8)
    # Feature 4: number of perturbations that significantly affect this gene
    sig_threshold = np.std(S) * 0.5
    feat_n_affected = np.sum(np.abs(S) > sig_threshold, axis=0).astype(np.float32)
    feat_n_affected = feat_n_affected / (np.max(feat_n_affected) + 1e-8)
    # Feature 5: was this gene itself perturbed? (binary)
    feat_is_perturbed = np.zeros(n_genes, dtype=np.float32)
    for idx in pert_gene_indices:
        feat_is_perturbed[idx] = 1.0

    gene_features = np.stack([
        feat_mean, feat_var, feat_response, feat_n_affected, feat_is_perturbed
    ], axis=1).astype(np.float32)

    return S, pert_gene_indices, pert_names, gene_features, pert_efficiency


# ── Stage 2: HYPHAE Model ───────────────────────────────────────────────────

class HYPHAE(nn.Module):
    """
    Perturbation-aware attention-based GRN inference model.
    Learns gene embeddings via self-attention, scores directed edges
    via an MLP, and trains by predicting perturbation effects through
    differentiable graph propagation.
    """

    def __init__(self, n_genes, n_features, d_model=128, n_heads=4,
                 n_layers=2, dropout=0.1, edge_chunk_size=256):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.edge_chunk_size = edge_chunk_size

        # Gene feature projection
        self.feature_proj = nn.Linear(n_features, d_model)

        # Learnable gene identity embedding
        self.gene_embed = nn.Embedding(n_genes, d_model)

        # Transformer encoder for gene-gene context sharing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Edge scoring MLP: (src_embed, tgt_embed) -> edge weight (signed scalar)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Small initialization for edge MLP to start near-zero adjacency
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def encode_genes(self, gene_features):
        """Produce contextualized gene embeddings via transformer."""
        feat_embed = self.feature_proj(gene_features)
        gene_ids = torch.arange(self.n_genes, device=gene_features.device)
        id_embed = self.gene_embed(gene_ids)
        x = feat_embed + id_embed
        x = x.unsqueeze(0)  # (1, n_genes, d_model) — single "batch"
        x = self.encoder(x)
        return x.squeeze(0)  # (n_genes, d_model)

    def score_edges(self, x):
        """
        Score all directed edges from gene embeddings.
        Uses chunked computation to stay within GPU memory.
        Returns A: (n_genes, n_genes), where A[t, r] = edge weight for r -> t.
        """
        n = x.shape[0]
        chunks = []
        for i in range(0, n, self.edge_chunk_size):
            i_end = min(i + self.edge_chunk_size, n)
            # Target genes in this chunk attend to all source (regulator) genes
            x_tgt = x[i:i_end].unsqueeze(1).expand(-1, n, -1)
            x_src = x.unsqueeze(0).expand(i_end - i, -1, -1)
            pair_embed = torch.cat([x_src, x_tgt], dim=-1)
            chunk_scores = self.edge_mlp(pair_embed).squeeze(-1)
            chunks.append(chunk_scores)
        A = torch.cat(chunks, dim=0)  # (n_genes, n_genes)
        # Zero out self-loops
        A = A * (1.0 - torch.eye(n, device=A.device))
        return A

    def forward(self, gene_features):
        """Full forward pass: features -> embeddings -> adjacency."""
        x = self.encode_genes(gene_features)
        A = self.score_edges(x)
        return A

    def predict_perturbation(self, A, pert_gene_indices, k_steps=1):
        """
        Predict expression changes under each perturbation via
        truncated k-step graph propagation.

        Args:
            A: (n_genes, n_genes) learned adjacency
            pert_gene_indices: list of gene indices that were perturbed
            k_steps: propagation depth (1 = direct effects only)

        Returns:
            predicted: (n_pert, n_genes) predicted expression changes
        """
        n = A.shape[0]
        n_pert = len(pert_gene_indices)
        device = A.device

        # Build perturbation input vectors: -1 at perturbed gene (CRISPRi knockdown)
        pert_vectors = torch.zeros(n, n_pert, device=device)
        for p, gene_idx in enumerate(pert_gene_indices):
            pert_vectors[gene_idx, p] = -1.0

        # Truncated k-step propagation
        current = pert_vectors
        accumulated = torch.zeros_like(pert_vectors)
        for _ in range(k_steps):
            current = A @ current
            accumulated = accumulated + current

        return accumulated.T  # (n_pert, n_genes)


# ── Stage 3: Training ────────────────────────────────────────────────────────

def train_hyphae(model, gene_features_t, S_train_t, S_val_t,
                 train_pert_indices, val_pert_indices,
                 n_epochs, lr, lambda_l1, k_steps, device,
                 patience=30):
    """
    Train HYPHAE model. Returns the best model state dict (by val loss).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        A = model(gene_features_t)

        # Training loss: predict perturbation effects
        pred_train = model.predict_perturbation(A, train_pert_indices, k_steps)
        mse_loss = F.mse_loss(pred_train, S_train_t)

        # L1 sparsity regularization on adjacency
        l1_loss = lambda_l1 * torch.abs(A).mean()

        total_loss = mse_loss + l1_loss
        total_loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            A_val = model(gene_features_t)
            pred_val = model.predict_perturbation(A_val, val_pert_indices, k_steps)
            val_loss = F.mse_loss(pred_val, S_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == n_epochs - 1:
            n_edges = (torch.abs(A) > 0.01).sum().item()
            print(f"  Epoch {epoch:4d} | train_mse={mse_loss.item():.6f} "
                  f"val_mse={val_loss:.6f} l1={l1_loss.item():.6f} "
                  f"edges(>0.01)={n_edges:,}", flush=True)

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    return best_state, best_val_loss


# ── Stage 4: Edge Extraction ─────────────────────────────────────────────────

def extract_edges(model, gene_features_t, gene_names, pert_names,
                  pert_efficiency, device, top_k=0):
    """Extract the final GRN edges from the trained model."""
    model.eval()
    with torch.no_grad():
        A = model(gene_features_t).cpu().numpy()

    n_genes = len(gene_names)
    records = []

    for t_idx in range(n_genes):
        for r_idx in range(n_genes):
            if t_idx == r_idx:
                continue
            weight = A[t_idx, r_idx]
            if abs(weight) < 1e-6:
                continue
            records.append({
                "Target": gene_names[t_idx],
                "Regulator": gene_names[r_idx],
                "Importance": abs(weight),
                "hyphae_weight": float(weight),
                "hyphae_sign": 1 if weight > 0 else -1,
                "pert_efficiency": pert_efficiency.get(gene_names[r_idx], 0.0),
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df

    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)

    if top_k > 0 and len(df) > top_k:
        df = df.head(top_k)

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HYPHAE — GPU GRN inference worker")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_id", type=int, required=True,
                        help="Bootstrap task ID (each task trains on a different cell subsample)")
    parser.add_argument("--total_tasks", type=int, required=True)
    parser.add_argument("--pert_col", type=str, default=None)
    parser.add_argument("--cell_subsample_frac", type=float, default=0.8)
    parser.add_argument("--efficiency_threshold", type=float, default=0.5)
    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--edge_chunk_size", type=int, default=256)
    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_l1", type=float, default=1e-4)
    parser.add_argument("--k_steps", type=int, default=1,
                        help="Propagation depth: 1=direct effects, 2=two-hop, etc.")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of perturbations held out for validation")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Keep top K edges per bootstrap (0 = keep all above threshold)")
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"bootstrap_{args.task_id}.parquet")
    if os.path.exists(out_file):
        print(f"Bootstrap {args.task_id} already complete. Skipping.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"HYPHAE GRN Worker — Bootstrap {args.task_id}/{args.total_tasks}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading: {args.input_file}")
    adata = ad.read_h5ad(args.input_file)
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)
    gene_names = list(adata.var_names)
    n_cells, n_genes = X.shape
    print(f"Shape: {X.shape}")

    # ── Detect perturbation labels ───────────────────────────────────────
    pert_col = args.pert_col or detect_perturbation_column(adata)
    if pert_col is None or pert_col not in adata.obs.columns:
        print("\nERROR: HYPHAE requires perturbation labels in adata.obs.")
        print("No perturbation column detected. Provide --pert_col explicitly.")
        print("HYPHAE cannot train without interventional data — exiting.")
        return

    control_mask = identify_control_mask(adata, pert_col)
    n_ctrl = int(control_mask.sum())
    n_pert = int((~control_mask).sum())
    print(f"Perturbation column: '{pert_col}' | {n_ctrl} control, {n_pert} perturbed cells")

    if n_ctrl < 20 or n_pert < 20:
        print("ERROR: Too few control or perturbed cells. Need at least 20 of each.")
        return

    # ── Bootstrap cell subsample ─────────────────────────────────────────
    rng = np.random.RandomState(42 + args.task_id)
    n_subsample = max(100, int(n_cells * args.cell_subsample_frac))
    cell_idx = rng.choice(n_cells, size=n_subsample, replace=False)
    X_sub = X[cell_idx]
    control_mask_sub = control_mask[cell_idx]
    pert_col_values_sub = adata.obs[pert_col].values[cell_idx]

    print(f"Bootstrap {args.task_id}: subsampled {n_subsample}/{n_cells} cells")

    # ── Stage 1: Compute perturbation signatures ────────────────────────
    t0 = time.time()
    S, pert_gene_indices, pert_names, gene_features, pert_efficiency = \
        compute_perturbation_signatures(
            X_sub, gene_names, pert_col_values_sub,
            control_mask_sub, min_cells=5,
            efficiency_threshold=args.efficiency_threshold,
        )
    n_pert_used = len(pert_names)
    print(f"Stage 1 complete: {n_pert_used} perturbations, "
          f"{n_genes} genes ({time.time() - t0:.1f}s)")

    if n_pert_used < 5:
        print("ERROR: Too few valid perturbations after filtering. Need at least 5.")
        return

    # ── Train/val split on perturbations ─────────────────────────────────
    n_val = max(1, int(n_pert_used * args.val_frac))
    n_train = n_pert_used - n_val
    perm = rng.permutation(n_pert_used)
    train_mask = perm[:n_train]
    val_mask = perm[n_train:]

    train_pert_indices = [pert_gene_indices[i] for i in train_mask]
    val_pert_indices = [pert_gene_indices[i] for i in val_mask]
    S_train = S[train_mask]
    S_val = S[val_mask]

    print(f"Perturbation split: {n_train} train, {n_val} val")

    # ── Move to device ───────────────────────────────────────────────────
    gene_features_t = torch.tensor(gene_features, dtype=torch.float32, device=device)
    S_train_t = torch.tensor(S_train, dtype=torch.float32, device=device)
    S_val_t = torch.tensor(S_val, dtype=torch.float32, device=device)

    # ── Stage 2+3: Build and train model ─────────────────────────────────
    model = HYPHAE(
        n_genes=n_genes,
        n_features=gene_features.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        edge_chunk_size=args.edge_chunk_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters | d_model={args.d_model} "
          f"heads={args.n_heads} layers={args.n_layers}")
    print(f"Training: {args.n_epochs} epochs, lr={args.lr}, "
          f"L1={args.lambda_l1}, k_steps={args.k_steps}")

    t0 = time.time()
    best_state, best_val_loss = train_hyphae(
        model, gene_features_t, S_train_t, S_val_t,
        train_pert_indices, val_pert_indices,
        n_epochs=args.n_epochs,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        k_steps=args.k_steps,
        device=device,
        patience=args.patience,
    )
    train_time = time.time() - t0
    print(f"Training complete: {train_time:.1f}s | best_val_mse={best_val_loss:.6f}")

    # ── Stage 4: Extract edges ───────────────────────────────────────────
    model.load_state_dict(best_state)
    model = model.to(device)

    df = extract_edges(
        model, gene_features_t, gene_names, pert_names,
        pert_efficiency, device, top_k=args.top_k,
    )

    df.to_parquet(out_file, index=False)
    print(f"Bootstrap {args.task_id}: saved {len(df):,} edges to {out_file}")
    print(f"Done. [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")


if __name__ == "__main__":
    main()
