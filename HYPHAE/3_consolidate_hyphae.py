"""
HYPHAE / 3_consolidate_hyphae.py
────────────────────────────────
Merges bootstrap parquets from HYPHAE array tasks into a single
rich GRN parquet. Computes stability, normalizes scores, and
produces output compatible with FUNGI and the GuanLab v2 schema.
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
from datetime import datetime


def normalize_column(series):
    vmin, vmax = series.min(), series.max()
    if vmax - vmin < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - vmin) / (vmax - vmin)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate HYPHAE bootstrap runs into a final GRN"
    )
    parser.add_argument("--bootstrap_dir", type=str, required=True,
                        help="Directory containing bootstrap_*.parquet files")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=0,
                        help="Keep top K edges (0 = keep all)")
    parser.add_argument("--min_stability", type=float, default=0.0,
                        help="Min bootstrap stability to retain edge")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"HYPHAE — Consolidation")
    print(f"{'='*70}")

    # ── Load all bootstrap files ─────────────────────────────────────────
    pattern = os.path.join(args.bootstrap_dir, "bootstrap_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No bootstrap_*.parquet files in {args.bootstrap_dir}")
        return

    n_bootstraps = len(files)
    print(f"Found {n_bootstraps} bootstrap files")

    # ── Aggregate edges across bootstraps ────────────────────────────────
    # For each edge, track: sum of weights, count of appearances, sign votes
    edge_data = {}

    for f in files:
        df = pd.read_parquet(f)
        print(f"  {os.path.basename(f)}: {len(df):,} edges")
        for _, row in df.iterrows():
            key = (row["Target"], row["Regulator"])
            if key not in edge_data:
                edge_data[key] = {
                    "weight_sum": 0.0,
                    "abs_weight_sum": 0.0,
                    "count": 0,
                    "sign_pos": 0,
                    "sign_neg": 0,
                    "pert_efficiency": row.get("pert_efficiency", 0.0),
                }
            d = edge_data[key]
            d["weight_sum"] += row["hyphae_weight"]
            d["abs_weight_sum"] += row["Importance"]
            d["count"] += 1
            if row["hyphae_sign"] > 0:
                d["sign_pos"] += 1
            else:
                d["sign_neg"] += 1

    print(f"\nUnique edges across all bootstraps: {len(edge_data):,}")

    # ── Build consolidated dataframe ─────────────────────────────────────
    records = []
    for (target, regulator), d in edge_data.items():
        stability = d["count"] / n_bootstraps
        avg_weight = d["weight_sum"] / d["count"]
        avg_importance = d["abs_weight_sum"] / d["count"]

        # Consensus sign: majority vote across bootstraps
        if d["sign_pos"] > d["sign_neg"]:
            consensus_sign = 1
        elif d["sign_neg"] > d["sign_pos"]:
            consensus_sign = -1
        else:
            consensus_sign = 0

        # Sign agreement: fraction of bootstraps that agree with consensus
        total_votes = d["sign_pos"] + d["sign_neg"]
        sign_agreement = max(d["sign_pos"], d["sign_neg"]) / total_votes

        records.append({
            "Target": target,
            "Regulator": regulator,
            "Importance": avg_importance,
            "hyphae_weight": avg_weight,
            "stability": stability,
            "consensus_sign": consensus_sign,
            "sign_agreement": sign_agreement,
            "pert_efficiency": d["pert_efficiency"],
            "n_bootstraps_present": d["count"],
        })

    df = pd.DataFrame(records)

    # ── Normalize ────────────────────────────────────────────────────────
    df["hyphae_confidence"] = normalize_column(df["Importance"])
    df["combined_score"] = (
        0.6 * df["hyphae_confidence"]
        + 0.4 * df["stability"]
    )

    # ── Filter ───────────────────────────────────────────────────────────
    if args.min_stability > 0:
        n_before = len(df)
        df = df[df["stability"] >= args.min_stability].copy()
        print(f"Stability filter (>={args.min_stability}): {n_before:,} -> {len(df):,}")

    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    if args.top_k > 0 and len(df) > args.top_k:
        df = df.head(args.top_k).copy()
        print(f"Top-K filter: kept {args.top_k:,} edges")

    df["rank"] = np.arange(1, len(df) + 1)

    # ── Reorder columns for output ───────────────────────────────────────
    output_cols = [
        "rank", "Target", "Regulator",
        "combined_score", "Importance", "hyphae_confidence",
        "hyphae_weight", "stability", "consensus_sign", "sign_agreement",
        "pert_efficiency", "n_bootstraps_present",
    ]
    df = df[[c for c in output_cols if c in df.columns]]

    # ── Save ─────────────────────────────────────────────────────────────
    df.to_parquet(args.output_file, index=False)
    print(f"\nFinal GRN saved to: {args.output_file}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"HYPHAE GRN Summary")
    print(f"{'─'*50}")
    print(f"  Total edges:           {len(df):,}")
    print(f"  Unique targets:        {df['Target'].nunique():,}")
    print(f"  Unique regulators:     {df['Regulator'].nunique():,}")
    print(f"  Bootstraps used:       {n_bootstraps}")

    n_pos = (df["consensus_sign"] == 1).sum()
    n_neg = (df["consensus_sign"] == -1).sum()
    print(f"  Activating edges (+):  {n_pos:,}")
    print(f"  Repressing edges (-):  {n_neg:,}")

    stab = df["stability"]
    print(f"  Stability — median:    {stab.median():.3f}")
    print(f"  Stability — mean:      {stab.mean():.3f}")
    print(f"  High-stability (>=0.8): {(stab >= 0.8).sum():,}")

    print(f"\nDone. [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")


if __name__ == "__main__":
    main()
