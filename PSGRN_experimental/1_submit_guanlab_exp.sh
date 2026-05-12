#!/bin/bash
#SBATCH --job-name=Your_Job_Name
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=90:00:00
#SBATCH --output=logs/guanlab_v2_%A_task_%a.log
#SBATCH --error=logs/guanlab_v2_%A_task_%a.err

# ── 1. CONTROL PANEL ────────────────────────────────────────────────────────
INPUT_H5AD=""

# GRN inference parameters
N_BOOTSTRAPS=20           # cell-subsampled bootstrap iterations (v1 used 10 seed-only)
CELL_SUBSAMPLE_FRAC=0.8   # fraction of cells per bootstrap
N_ESTIMATORS=500           # max LightGBM boosting rounds (early stopping at 30)
EFFICIENCY_THRESHOLD=0.5   # min knockdown efficiency in SD to include a perturbation
# Set PERT_COL to auto-detect, or specify explicitly (e.g., "perturbation")
PERT_COL=""

# ── 2. DIRECTORY SETUP ──────────────────────────────────────────────────────
BASE_DIR=""
CHUNK_DIR="${BASE_DIR}/chunks_"
LOG_DIR="${BASE_DIR}/logs/GuanLab_Exp"

mkdir -p "$CHUNK_DIR"
mkdir -p "$LOG_DIR"

# ── 3. ENVIRONMENT ──────────────────────────────────────────────────────────
PYTHON_BIN=/scratch/patrick.sheehan/FUNGI_bot/bin/python

# Sanity checks
$PYTHON_BIN -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')" \
    || { echo "CRITICAL: LightGBM not found in $PYTHON_BIN"; exit 1; }
$PYTHON_BIN -c "from scipy.stats import pearsonr; print('scipy OK')" \
    || { echo "CRITICAL: scipy not found in $PYTHON_BIN"; exit 1; }

# ── 4. EXECUTION ────────────────────────────────────────────────────────────
echo "=================================================================="
echo "GuanLab GRN v2 — Array Task: $SLURM_ARRAY_TASK_ID"
echo "Input:  $INPUT_H5AD"
echo "Output: $CHUNK_DIR"
echo "Bootstraps: $N_BOOTSTRAPS | Subsample: $CELL_SUBSAMPLE_FRAC"
echo "=================================================================="

cd "$BASE_DIR"

PERT_FLAG=""
if [ -n "$PERT_COL" ]; then
    PERT_FLAG="--pert_col $PERT_COL"
fi

srun $PYTHON_BIN 2_guanlab_exp.py \
    --input_file "$INPUT_H5AD" \
    --output_dir "$CHUNK_DIR" \
    --task_id $SLURM_ARRAY_TASK_ID \
    --total_tasks 10 \
    --n_jobs 32 \
    --n_bootstraps $N_BOOTSTRAPS \
    --cell_subsample_frac $CELL_SUBSAMPLE_FRAC \
    --n_estimators $N_ESTIMATORS \
    --efficiency_threshold $EFFICIENCY_THRESHOLD \
    $PERT_FLAG

echo "Task $SLURM_ARRAY_TASK_ID complete."
