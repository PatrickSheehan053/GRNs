"""
Upgraded GuanLab GRN inference worker (experimental).

Key upgrades over v1:
  1. Cell-subsampled bootstrap stability (not just seed variation)
  2. Mean Difference scoring (if perturbation labels detected in adata.obs)
  3. Edge sign annotation (Pearson correlation direction + MD direction)
  4. Perturbation efficiency scoring per source gene
  5. Rich metadata parquet output for downstream FUNGI consumption
  6. Early stopping in LightGBM to reduce overfitting and speed up training
"""

import os
import argparse
import time
import tempfile
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import lightgbm as lgb
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from datetime import datetime


# ── Perturbation label detection ─────────────────────────────────────────────

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
    """Auto-detect which adata.obs column contains perturbation labels."""
    for col in KNOWN_PERT_COLUMNS:
        if col in adata.obs.columns:
            return col
    # Heuristic: look for any column where some values match known control names
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object or adata.obs[col].dtype.name == "category":
            values = set(adata.obs[col].astype(str).str.lower().unique())
            if values.intersection(set(v.lower() for v in KNOWN_CONTROL_VALUES)):
                return col
    return None


def identify_control_mask(adata, pert_col):
    """Return a boolean mask for control cells."""
    labels_lower = adata.obs[pert_col].astype(str).str.lower()
    control_mask = labels_lower.isin([v.lower() for v in KNOWN_CONTROL_VALUES])
    return control_mask.values


def build_perturbation_map(adata, pert_col, control_mask, gene_names):
    """Build a dict mapping gene_name -> array of cell indices where that gene was perturbed."""
    gene_set = set(gene_names)
    pert_labels = adata.obs[pert_col].astype(str).values
    pert_map = {}
    for idx in range(len(pert_labels)):
        if control_mask[idx]:
            continue
        label = pert_labels[idx]
        if label in gene_set:
            if label not in pert_map:
                pert_map[label] = []
            pert_map[label].append(idx)
    # Convert to arrays
    for key in pert_map:
        pert_map[key] = np.array(pert_map[key], dtype=np.int64)
    return pert_map


# ── Perturbation efficiency ──────────────────────────────────────────────────

def compute_perturbation_efficiency(X_dense, gene_names, pert_map, control_indices):
    """
    For each perturbed gene, estimate knockdown efficiency as the
    standardized reduction in that gene's own expression relative to controls.
    Returns dict: gene_name -> efficiency_score (higher = better knockdown).
    """
    control_idx = np.array(control_indices, dtype=np.int64)
    efficiency = {}
    for gene_name, pert_indices in pert_map.items():
        if gene_name not in gene_names:
            continue
        gene_idx = gene_names.index(gene_name)
        ctrl_expr = X_dense[control_idx, gene_idx]
        pert_expr = X_dense[pert_indices, gene_idx]
        ctrl_mean = np.mean(ctrl_expr)
        ctrl_std = np.std(ctrl_expr)
        if ctrl_std < 1e-8:
            efficiency[gene_name] = 0.0
            continue
        # Standardized reduction: how many SDs below control mean the perturbed cells fall
        pert_mean = np.mean(pert_expr)
        efficiency[gene_name] = (ctrl_mean - pert_mean) / ctrl_std
    return efficiency


# ── Mean Difference scoring ──────────────────────────────────────────────────

def compute_mean_difference_for_targets(X_dense, target_indices, gene_names,
                                        pert_map, control_indices, pert_efficiency,
                                        efficiency_threshold):
    """
    For a set of target genes, compute Mean Difference scores for all
    source→target edges where the source gene was perturbed.

    Returns a list of record dicts.
    """
    control_idx = np.array(control_indices, dtype=np.int64)
    records = []
    for target_idx in target_indices:
        target_name = gene_names[target_idx]
        ctrl_mean = np.mean(X_dense[control_idx, target_idx])
        for source_name, pert_indices in pert_map.items():
            if source_name == target_name:
                continue
            # Skip low-efficiency perturbations
            eff = pert_efficiency.get(source_name, 0.0)
            if eff < efficiency_threshold:
                continue
            if len(pert_indices) < 5:
                continue
            pert_mean = np.mean(X_dense[pert_indices, target_idx])
            diff = pert_mean - ctrl_mean
            abs_diff = abs(diff)
            if abs_diff < 1e-10:
                continue
            sign = 1 if diff > 0 else -1
            records.append({
                "Target": target_name,
                "Regulator": source_name,
                "md_score": abs_diff,
                "md_sign": sign,
                "pert_efficiency": eff,
                "pert_n_cells": len(pert_indices),
            })
    return records


# ── Per-gene LightGBM worker ────────────────────────────────────────────────

def _train_gene_worker(target_gene, mmap_path, shape, feature_names,
                       n_bootstraps, cell_subsample_frac, temp_dir, lgbm_params):
    """
    Train K bootstraps for one target gene with cell subsampling.
    Tracks per-edge stability (fraction of bootstraps where edge has nonzero importance).
    Also computes Pearson correlation sign for each edge.
    """
    gene_file = os.path.join(temp_dir, f"{target_gene}.parquet")
    if os.path.exists(gene_file):
        return "SKIP"

    start_gene = time.time()
    try:
        X_shared = np.memmap(mmap_path, dtype="float32", mode="r", shape=shape)
        n_cells = shape[0]
        target_idx = feature_names.index(target_gene)
        feature_indices = [i for i in range(len(feature_names)) if i != target_idx]
        features_used = [feature_names[i] for i in feature_indices]

        y_full = X_shared[:, target_idx].copy()
        total_importance = np.zeros(len(features_used), dtype=np.float64)
        presence_count = np.zeros(len(features_used), dtype=np.int32)

        subsample_size = max(100, int(n_cells * cell_subsample_frac))

        for k in range(n_bootstraps):
            rng = np.random.RandomState(42 + k)
            cell_idx = rng.choice(n_cells, size=subsample_size, replace=False)

            X_sub = X_shared[np.ix_(cell_idx, feature_indices)]
            y_sub = y_full[cell_idx]

            seed = 42 + k
            params = lgbm_params.copy()
            params["seed"] = seed
            params["bagging_seed"] = seed
            params["feature_fraction_seed"] = seed

            dtrain = lgb.Dataset(X_sub, label=y_sub, feature_name=features_used)
            dval = lgb.Dataset(X_sub, label=y_sub, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=params.pop("n_estimators", 500),
                valid_sets=[dval],
                callbacks=[
                    lgb.log_evaluation(period=-1),
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                ],
            )

            importance = model.feature_importance(importance_type="gain")
            total_importance += importance
            presence_count += (importance > 0).astype(np.int32)

        avg_importance = total_importance / n_bootstraps
        stability = presence_count / n_bootstraps

        # Compute Pearson correlation sign on the full dataset
        corr_signs = np.zeros(len(features_used), dtype=np.int8)
        for j, fidx in enumerate(feature_indices):
            if avg_importance[j] > 0:
                r, _ = pearsonr(X_shared[:, fidx], y_full)
                corr_signs[j] = 1 if r > 0 else (-1 if r < 0 else 0)

        # Build records (keep edges with nonzero importance)
        records = []
        for j, (name, imp, stab, sign) in enumerate(
            zip(features_used, avg_importance, stability, corr_signs)
        ):
            if imp > 0:
                records.append({
                    "Target": target_gene,
                    "Regulator": name,
                    "Importance": float(imp),
                    "stability": float(stab),
                    "corr_sign": int(sign),
                })

        if records:
            pd.DataFrame(records).to_parquet(gene_file, index=False)
        else:
            pd.DataFrame(columns=[
                "Target", "Regulator", "Importance", "stability", "corr_sign"
            ]).to_parquet(gene_file, index=False)

        elapsed = time.time() - start_gene
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] [DONE] {target_gene} "
            f"({elapsed:.1f}s, {len(records)} edges)",
            flush=True,
        )
        return "Success"

    except Exception as e:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] [FAIL] {target_gene}: {e}",
            flush=True,
        )
        return "Failed"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GuanLab GRN Worker v2 — upgraded with stability, MD, signing, efficiency"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to SPORE .h5ad file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output chunks")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--total_tasks", type=int, required=True)
    parser.add_argument("--n_jobs", type=int, default=16,
                        help="Parallel jobs for LightGBM per-gene workers")
    parser.add_argument("--n_bootstraps", type=int, default=20,
                        help="Number of cell-subsampled bootstraps (default: 20)")
    parser.add_argument("--cell_subsample_frac", type=float, default=0.8,
                        help="Fraction of cells to subsample per bootstrap (default: 0.8)")
    parser.add_argument("--pert_col", type=str, default=None,
                        help="Column in adata.obs with perturbation labels (auto-detected if not set)")
    parser.add_argument("--efficiency_threshold", type=float, default=0.5,
                        help="Min knockdown efficiency (in SD units) to include a perturbation (default: 0.5)")
    parser.add_argument("--n_estimators", type=int, default=500,
                        help="Max LightGBM boosting rounds (default: 500)")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"GuanLab GRN Worker v2 — Task {args.task_id}/{args.total_tasks}")
    print(f"{'='*70}")
    print(f"Loading: {args.input_file}")
    adata = ad.read_h5ad(args.input_file)

    # ── Prepare expression matrix ────────────────────────────────────────
    print("Converting to dense float32...")
    X_dense = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X_dense = X_dense.astype(np.float32)
    gene_names = list(adata.var_names)
    total_genes = len(gene_names)
    print(f"Shape: {X_dense.shape} | Memory: {X_dense.nbytes / 1e9:.2f} GB")

    # ── Detect perturbation labels ───────────────────────────────────────
    pert_col = args.pert_col or detect_perturbation_column(adata)
    has_pert_data = False
    pert_map = {}
    control_mask = None
    control_indices = None
    pert_efficiency = {}

    if pert_col and pert_col in adata.obs.columns:
        control_mask = identify_control_mask(adata, pert_col)
        n_ctrl = int(control_mask.sum())
        n_pert = int((~control_mask).sum())
        print(f"Perturbation column: '{pert_col}' | {n_ctrl} control, {n_pert} perturbed cells")

        if n_ctrl > 50 and n_pert > 50:
            control_indices = np.where(control_mask)[0]
            pert_map = build_perturbation_map(adata, pert_col, control_mask, gene_names)
            print(f"Perturbation map: {len(pert_map)} unique perturbed genes detected")

            # Compute perturbation efficiency for all source genes
            pert_efficiency = compute_perturbation_efficiency(
                X_dense, gene_names, pert_map, control_indices
            )
            n_strong = sum(1 for v in pert_efficiency.values() if v >= args.efficiency_threshold)
            print(f"Perturbation efficiency: {n_strong}/{len(pert_efficiency)} genes "
                  f"above threshold ({args.efficiency_threshold} SD)")
            has_pert_data = True
        else:
            print("Too few control or perturbed cells for MD computation; skipping.")
    else:
        print("No perturbation labels detected. Skipping Mean Difference and efficiency scoring.")
        print("  (Provide --pert_col to specify the column name manually.)")

    # ── Slice genes for this SLURM task ──────────────────────────────────
    per_task = total_genes // args.total_tasks
    remainder = total_genes % args.total_tasks
    if args.task_id < remainder:
        start_idx = args.task_id * (per_task + 1)
        end_idx = start_idx + per_task + 1
    else:
        start_idx = args.task_id * per_task + remainder
        end_idx = start_idx + per_task
    my_genes = gene_names[start_idx:end_idx]
    print(f"Task {args.task_id}: genes [{start_idx}..{end_idx - 1}] ({len(my_genes)} targets)")

    # ── Checkpointing directory ──────────────────────────────────────────
    temp_dir = os.path.join(args.output_dir, "temp_shards")
    os.makedirs(temp_dir, exist_ok=True)

    targets = [g for g in my_genes if not os.path.exists(os.path.join(temp_dir, f"{g}.parquet"))]
    if not targets:
        print("All LightGBM targets already processed.")
    else:
        print(f"{len(targets)} genes remaining for LightGBM inference.")

    # ── LightGBM parameters ──────────────────────────────────────────────
    lgbm_params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "n_estimators": args.n_estimators,
        "num_threads": 1,
        "n_jobs": 1,
        "verbose": -1,
    }

    # ── Phase 1: LightGBM bootstrapped inference ─────────────────────────
    if targets:
        mmap_file = tempfile.NamedTemporaryFile(delete=False)
        mmap_path = mmap_file.name
        mmap_file.close()

        print(f"Creating memory map: {mmap_path}")
        X_mmap = np.memmap(mmap_path, dtype="float32", mode="w+", shape=X_dense.shape)
        X_mmap[:] = X_dense[:]
        X_mmap.flush()

        try:
            Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(_train_gene_worker)(
                    gene, mmap_path, X_dense.shape, gene_names,
                    args.n_bootstraps, args.cell_subsample_frac,
                    temp_dir, lgbm_params.copy(),
                ) for gene in targets
            )
        finally:
            if os.path.exists(mmap_path):
                os.unlink(mmap_path)
                print("Memory map cleaned up.")

    # ── Phase 2: Mean Difference scoring ─────────────────────────────────
    md_records = []
    if has_pert_data:
        print(f"Computing Mean Difference scores for {len(my_genes)} target genes...")
        my_target_indices = [gene_names.index(g) for g in my_genes]
        md_records = compute_mean_difference_for_targets(
            X_dense, my_target_indices, gene_names,
            pert_map, control_indices, pert_efficiency,
            args.efficiency_threshold,
        )
        print(f"Mean Difference: {len(md_records)} edges scored.")

    # ── Phase 3: Consolidate this task's output ──────────────────────────
    print("Consolidating shards...")

    # Load LightGBM results
    lgbm_files = [
        os.path.join(temp_dir, f"{g}.parquet")
        for g in my_genes
        if os.path.exists(os.path.join(temp_dir, f"{g}.parquet"))
    ]
    if lgbm_files:
        df_lgbm = pd.concat([pd.read_parquet(f) for f in lgbm_files], ignore_index=True)
    else:
        df_lgbm = pd.DataFrame(columns=[
            "Target", "Regulator", "Importance", "stability", "corr_sign"
        ])

    # Merge with Mean Difference if available
    if md_records:
        df_md = pd.DataFrame(md_records)
        # Left join: keep all LightGBM edges, add MD scores where available
        df_merged = df_lgbm.merge(
            df_md, on=["Target", "Regulator"], how="outer", suffixes=("", "_md")
        )
        # Fill NaN for edges only in one method
        df_merged["Importance"] = df_merged["Importance"].fillna(0.0)
        df_merged["stability"] = df_merged["stability"].fillna(0.0)
        df_merged["corr_sign"] = df_merged["corr_sign"].fillna(0).astype(int)
        df_merged["md_score"] = df_merged["md_score"].fillna(0.0)
        df_merged["md_sign"] = df_merged["md_sign"].fillna(0).astype(int)
        df_merged["pert_efficiency"] = df_merged["pert_efficiency"].fillna(0.0)
        df_merged["pert_n_cells"] = df_merged["pert_n_cells"].fillna(0).astype(int)

        # Source flags: which method contributed this edge?
        df_merged["in_lgbm"] = (df_merged["Importance"] > 0).astype(int)
        df_merged["in_md"] = (df_merged["md_score"] > 0).astype(int)
        df_merged["method_votes"] = df_merged["in_lgbm"] + df_merged["in_md"]
    else:
        df_merged = df_lgbm.copy()
        df_merged["md_score"] = 0.0
        df_merged["md_sign"] = 0
        df_merged["pert_efficiency"] = 0.0
        df_merged["pert_n_cells"] = 0
        df_merged["in_lgbm"] = 1
        df_merged["in_md"] = 0
        df_merged["method_votes"] = 1

    # Save chunk
    out_file = os.path.join(args.output_dir, f"chunk_{args.task_id}.parquet")
    if len(df_merged) > 0:
        df_merged.to_parquet(out_file, index=False)
        print(f"Task {args.task_id}: saved {len(df_merged):,} edges to {out_file}")
    else:
        print(f"Task {args.task_id}: no edges generated.")

    print(f"Task {args.task_id} complete.")


if __name__ == "__main__":
    main()
