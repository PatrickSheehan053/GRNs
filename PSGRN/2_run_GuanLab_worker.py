"""
to_guanlab / 2_run_guanlab_worker.py
────────────────────────────────────
Worker node for GuanLab parent graph generation.
Fully self-contained. Reads a SPORE .h5ad, memory-maps the matrix,
and runs parallel LightGBM bootstrapping to extract edge importances.
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
from joblib import Parallel, delayed
from datetime import datetime

def _train_gene_worker(target_gene, mmap_path, shape, feature_names, n_bootstraps, temp_dir, lgbm_params):
    """Isolated worker process: Trains K bootstraps for one target gene."""
    gene_file = os.path.join(temp_dir, f"{target_gene}.parquet")
    
    if os.path.exists(gene_file):
        return "SKIP"
    
    start_gene = time.time()
    try:
        # Load shared memory (read-only) to prevent RAM explosion across 32 cores
        X_shared = np.memmap(mmap_path, dtype='float32', mode='r', shape=shape)
        
        target_idx = feature_names.index(target_gene)
        
        # Create feature matrix (all columns EXCEPT the target gene)
        feature_indices = [i for i in range(len(feature_names)) if i != target_idx]
        X = X_shared[:, feature_indices]
        y = X_shared[:, target_idx]
        
        features_used = [feature_names[i] for i in feature_indices]
        total_importance = np.zeros(len(features_used))
        
        # K-Bootstrap Loop
        for k in range(n_bootstraps):
            seed = 42 + k
            params = lgbm_params.copy()
            params['seed'] = seed
            params['bagging_seed'] = seed
            params['feature_fraction_seed'] = seed
            
            dtrain = lgb.Dataset(X, label=y, feature_name=features_used)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                valid_sets=[dtrain],
                callbacks=[lgb.log_evaluation(period=-1)]
            )
            
            total_importance += model.feature_importance(importance_type='gain')
        
        # Average importances across the K bootstraps
        avg_importance = total_importance / n_bootstraps
        
        # Format records (filter out zero importances to save disk space)
        records = [
            {'Target': target_gene, 'Regulator': name, 'Importance': score}
            for name, score in zip(features_used, avg_importance) if score > 0
        ]
        
        if records:
            pd.DataFrame(records).to_parquet(gene_file)
        else:
            pd.DataFrame({'Target': [], 'Regulator': [], 'Importance': []}).to_parquet(gene_file)
        
        elapsed = time.time() - start_gene
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DONE] {target_gene} ({elapsed:.1f}s)", flush=True)
        return "Success"
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [FAIL] {target_gene}: {str(e)}", flush=True)
        return "Failed"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--total_tasks", type=int, required=True)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading SPORE dataset: {args.input_file}")
    adata = ad.read_h5ad(args.input_file)
    
    # 1. Prepare Data
    print("Converting to dense float32 array...")
    X_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    X_dense = X_dense.astype(np.float32)
    gene_names = list(adata.var_names)
    total_genes = len(gene_names)
    
    print(f"Data shape: {X_dense.shape} | Memory footprint: {X_dense.nbytes / 1e9:.2f} GB")
    
    # 2. Slice the genes based on SLURM array ID
    per_task = total_genes // args.total_tasks
    remainder = total_genes % args.total_tasks

    if args.task_id < remainder:
        start_idx = args.task_id * (per_task + 1)
        end_idx   = start_idx + per_task + 1
    else:
        start_idx = args.task_id * per_task + remainder
        end_idx   = start_idx + per_task

    my_genes = gene_names[start_idx:end_idx]
    print(f"Task {args.task_id}: Processing genes {start_idx} to {end_idx-1} ({len(my_genes)} targets)")
    
    # 3. Setup Temp Checkpoint Directory
    temp_dir = os.path.join(args.output_dir, "temp_shards")
    os.makedirs(temp_dir, exist_ok=True)
    
    targets = [g for g in my_genes if not os.path.exists(os.path.join(temp_dir, f"{g}.parquet"))]
    if not targets:
        print("All assigned genes already processed. Consolidating existing shards...")
    
    # 4. LightGBM Parameters
    lgbm_params = {
        'objective': 'regression', 'metric': 'mse', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'min_data_in_leaf': 10,
        'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'max_depth': -1, 'n_estimators': 500, 'num_threads': 1, 'n_jobs': 1, 'verbose': -1
    }

    # 5. Execute with Shared Memory Map
    if targets:
        mmap_file = tempfile.NamedTemporaryFile(delete=False)
        mmap_path = mmap_file.name
        mmap_file.close()
        
        print(f"Creating shared memory mapping: {mmap_path}")
        X_shared = np.memmap(mmap_path, dtype='float32', mode='w+', shape=X_dense.shape)
        X_shared[:] = X_dense[:]
        X_shared.flush()
        
        try:
            Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(_train_gene_worker)(
                    gene, mmap_path, X_dense.shape, gene_names,
                    10, temp_dir, lgbm_params
                ) for gene in targets
            )
        finally:
            if os.path.exists(mmap_path):
                os.unlink(mmap_path)
                print("Cleaned up shared memory.")
    
    # 6. Consolidate This Task's Shards
    out_file = os.path.join(args.output_dir, f"chunk_{args.task_id}.parquet")
    my_files = [os.path.join(temp_dir, f"{g}.parquet") for g in my_genes if os.path.exists(os.path.join(temp_dir, f"{g}.parquet"))]
    
    if my_files:
        df = pd.concat([pd.read_parquet(f) for f in my_files], ignore_index=True)
        df.to_parquet(out_file)
        print(f"Task {args.task_id} successfully saved to {out_file} ({len(df):,} edges)")
    else:
        print(f"Task {args.task_id} failed to generate any data.")

if __name__ == "__main__":
    main()
