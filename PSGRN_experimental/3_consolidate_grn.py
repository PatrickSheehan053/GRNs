"""
to_guanlab / 3_consolidate_grn.py
─────────────────────────────────
Post-processing consolidation script for GuanLab GRN experimental.

Runs ONCE after all SLURM array workers finish. Does:
  1. Merges all chunk_*.parquet files into one edge table
  2. Normalizes LightGBM importance → lgbm_confidence ∈ [0, 1]
  3. Normalizes Mean Difference scores → md_confidence ∈ [0, 1]
  4. Computes a weighted combined_score fusing both methods
  5. Assigns a consensus edge sign from correlation and MD directions
  6. Exports the final rich GRN parquet ready for FUNGI ingestion

Usage:
    python 3_consolidate_grn.py \
        --chunk_dir /path/to/chunks_REP_mc \
        --output_file /path/to/grn_v2_final.parquet \
        --top_k 0           # 0 = keep all edges (default)
        --lgbm_weight 0.5   # weight for LightGBM in combined score
        --md_weight 0.3     # weight for Mean Difference
        --stability_weight 0.2  # weight for bootstrap stability
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
from datetime import datetime


def normalize_column(series):
    """Min-max normalize a series to [0, 1]. Returns zeros if range is zero."""
    vmin = series.min()
    vmax = series.max()
    if vmax - vmin < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - vmin) / (vmax - vmin)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate GuanLab GRN v2 chunk parquets into a final rich GRN"
    )
    parser.add_argument("--chunk_dir", type=str, required=True,
                        help="Directory containing chunk_*.parquet files")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path for the final consolidated GRN parquet")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Keep only top K edges by combined_score (0 = keep all)")
    parser.add_argument("--lgbm_weight", type=float, default=0.5,
                        help="Weight for LightGBM confidence in combined score")
    parser.add_argument("--md_weight", type=float, default=0.3,
                        help="Weight for Mean Difference confidence in combined score")
    parser.add_argument("--stability_weight", type=float, default=0.2,
                        help="Weight for bootstrap stability in combined score")
    parser.add_argument("--min_stability", type=float, default=0.0,
                        help="Minimum bootstrap stability to retain an edge (0 = no filter)")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"GuanLab GRN v2 — Consolidation")
    print(f"{'='*70}")

    # ── 1. Load and merge all chunks ─────────────────────────────────────
    chunk_pattern = os.path.join(args.chunk_dir, "chunk_*.parquet")
    chunk_files = sorted(glob.glob(chunk_pattern))
    if not chunk_files:
        print(f"ERROR: No chunk_*.parquet files found in {args.chunk_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files")
    frames = []
    for f in chunk_files:
        df = pd.read_parquet(f)
        frames.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} edges")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal raw edges after merge: {len(df):,}")

    # Handle potential duplicates from outer-join in workers (same edge from
    # different chunks shouldn't happen, but guard against restarts)
    n_before = len(df)
    df = df.groupby(["Target", "Regulator"], as_index=False).first()
    if len(df) < n_before:
        print(f"Removed {n_before - len(df):,} duplicate edges")

    # ── 2. Ensure all expected columns exist ─────────────────────────────
    for col, default in [
        ("Importance", 0.0), ("stability", 0.0), ("corr_sign", 0),
        ("md_score", 0.0), ("md_sign", 0), ("pert_efficiency", 0.0),
        ("pert_n_cells", 0), ("in_lgbm", 0), ("in_md", 0), ("method_votes", 0),
    ]:
        if col not in df.columns:
            df[col] = default

    # ── 3. Normalize scores to [0, 1] confidence ────────────────────────
    df["lgbm_confidence"] = normalize_column(df["Importance"])
    df["md_confidence"] = normalize_column(df["md_score"])

    # ── 4. Compute consensus edge sign ───────────────────────────────────
    # Priority: if MD sign is available, use it (it's derived from actual
    # perturbation effect direction). Fall back to correlation sign.
    def consensus_sign(row):
        if row["md_sign"] != 0:
            return int(row["md_sign"])
        return int(row["corr_sign"])

    df["consensus_sign"] = df.apply(consensus_sign, axis=1)

    # Sign agreement flag: do correlation and MD agree on direction?
    has_both_signs = (df["corr_sign"] != 0) & (df["md_sign"] != 0)
    df["sign_agreement"] = 0
    df.loc[has_both_signs, "sign_agreement"] = (
        (df.loc[has_both_signs, "corr_sign"] == df.loc[has_both_signs, "md_sign"]).astype(int)
    )

    # ── 5. Compute combined score ────────────────────────────────────────
    w_lgbm = args.lgbm_weight
    w_md = args.md_weight
    w_stab = args.stability_weight
    total_w = w_lgbm + w_md + w_stab
    # Normalize weights so they sum to 1
    w_lgbm /= total_w
    w_md /= total_w
    w_stab /= total_w

    df["combined_score"] = (
        w_lgbm * df["lgbm_confidence"]
        + w_md * df["md_confidence"]
        + w_stab * df["stability"]
    )

    # ── 6. Apply stability filter if requested ───────────────────────────
    if args.min_stability > 0:
        n_before = len(df)
        # Only filter edges that came from LightGBM (MD-only edges have stability=0)
        drop_mask = (df["in_lgbm"] == 1) & (df["stability"] < args.min_stability)
        df = df[~drop_mask].copy()
        print(f"Stability filter (>= {args.min_stability}): "
              f"{n_before:,} → {len(df):,} edges")

    # ── 7. Sort and optionally truncate ──────────────────────────────────
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    if args.top_k > 0 and len(df) > args.top_k:
        df = df.head(args.top_k).copy()
        print(f"Top-K filter: kept top {args.top_k:,} edges")

    # ── 8. Add rank column ───────────────────────────────────────────────
    df["rank"] = np.arange(1, len(df) + 1)

    # ── 9. Select and order final columns ────────────────────────────────
    output_columns = [
        "rank", "Target", "Regulator",
        # Scores
        "combined_score", "Importance", "lgbm_confidence",
        "md_score", "md_confidence",
        # Stability & signs
        "stability", "consensus_sign", "corr_sign", "md_sign", "sign_agreement",
        # Perturbation metadata
        "pert_efficiency", "pert_n_cells",
        # Method provenance
        "in_lgbm", "in_md", "method_votes",
    ]
    # Only include columns that actually exist
    output_columns = [c for c in output_columns if c in df.columns]
    df = df[output_columns]

    # ── 10. Save ─────────────────────────────────────────────────────────
    df.to_parquet(args.output_file, index=False)
    print(f"\nFinal GRN saved to: {args.output_file}")

    # ── Summary statistics ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"GRN Summary")
    print(f"{'─'*50}")
    print(f"  Total edges:           {len(df):,}")
    print(f"  Unique targets:        {df['Target'].nunique():,}")
    print(f"  Unique regulators:     {df['Regulator'].nunique():,}")

    n_lgbm = (df["in_lgbm"] == 1).sum()
    n_md = (df["in_md"] == 1).sum()
    n_both = (df["method_votes"] == 2).sum()
    print(f"  LightGBM-only edges:   {n_lgbm - n_both:,}")
    print(f"  MeanDiff-only edges:   {n_md - n_both:,}")
    print(f"  Concordant (both):     {n_both:,}")

    n_positive = (df["consensus_sign"] == 1).sum()
    n_negative = (df["consensus_sign"] == -1).sum()
    n_unsigned = (df["consensus_sign"] == 0).sum()
    print(f"  Activating edges (+):  {n_positive:,}")
    print(f"  Repressing edges (-):  {n_negative:,}")
    print(f"  Unsigned edges:        {n_unsigned:,}")

    if (df["stability"] > 0).any():
        stab_vals = df.loc[df["stability"] > 0, "stability"]
        print(f"  Stability — median:    {stab_vals.median():.3f}")
        print(f"  Stability — mean:      {stab_vals.mean():.3f}")
        high_stab = (stab_vals >= 0.8).sum()
        print(f"  High-stability (≥0.8): {high_stab:,}")

    print(f"\nDone. [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")


if __name__ == "__main__":
    main()
