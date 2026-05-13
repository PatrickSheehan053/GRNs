#!/bin/bash
#SBATCH --job-name=HYPHAE
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hyphae_%A_boot_%a.log
#SBATCH --error=logs/hyphae_%A_boot_%a.err
#SBATCH --partition=gpuq

# ── 1. CONTROL PANEL ────────────────────────────────────────────────────────
INPUT_H5AD="path/to/your/processed/training/h5ad/file"

# Bootstrap parameters
TOTAL_BOOTSTRAPS=20        # must match --array range above
CELL_SUBSAMPLE_FRAC=0.8
EFFICIENCY_THRESHOLD=0.5
PERT_COL=""                # auto-detect if empty

# Model hyperparameters
D_MODEL=128
N_HEADS=4
N_LAYERS=2
DROPOUT=0.1
EDGE_CHUNK_SIZE=256        # reduce to 128 if V100 runs out of memory

# Training hyperparameters
N_EPOCHS=300
LR=0.001
LAMBDA_L1=0.0001
K_STEPS=1                 # 1 = direct effects only (recommended start)
VAL_FRAC=0.2
PATIENCE=30

# ── 2. DIRECTORY SETUP ──────────────────────────────────────────────────────
BASE_DIR="path/to/your/base/directory"
OUTPUT_DIR="${BASE_DIR}/bootstraps_"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ── 3. ENVIRONMENT ──────────────────────────────────────────────────────────
PYTHON_BIN=/path/to/your/venv

# Sanity checks
$PYTHON_BIN -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" \
    || { echo "CRITICAL: PyTorch not found or no CUDA"; exit 1; }
$PYTHON_BIN -c "import anndata; print(f'AnnData: {anndata.__version__}')" \
    || { echo "CRITICAL: anndata not found"; exit 1; }

# ── 4. EXECUTION ────────────────────────────────────────────────────────────
echo "=================================================================="
echo "HYPHAE GRN — Bootstrap: $SLURM_ARRAY_TASK_ID / $TOTAL_BOOTSTRAPS"
echo "Input:  $INPUT_H5AD"
echo "Output: $OUTPUT_DIR"
echo "=================================================================="

cd "$BASE_DIR"

PERT_FLAG=""
if [ -n "$PERT_COL" ]; then
    PERT_FLAG="--pert_col $PERT_COL"
fi

srun $PYTHON_BIN 2_run_hyphae_worker.py \
    --input_file "$INPUT_H5AD" \
    --output_dir "$OUTPUT_DIR" \
    --task_id $SLURM_ARRAY_TASK_ID \
    --total_tasks $TOTAL_BOOTSTRAPS \
    --cell_subsample_frac $CELL_SUBSAMPLE_FRAC \
    --efficiency_threshold $EFFICIENCY_THRESHOLD \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --edge_chunk_size $EDGE_CHUNK_SIZE \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --lambda_l1 $LAMBDA_L1 \
    --k_steps $K_STEPS \
    --val_frac $VAL_FRAC \
    --patience $PATIENCE \
    $PERT_FLAG

echo "Bootstrap $SLURM_ARRAY_TASK_ID complete."
