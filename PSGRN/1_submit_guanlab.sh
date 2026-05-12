#!/bin/bash
#SBATCH --job-name=VCC_CHITIN
#SBATCH --array=0-9                 
#SBATCH --nodes=1                   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          
#SBATCH --mem=512G                  
#SBATCH --time=90:00:00
#SBATCH --output=logs/guanlab_%A_task_%a.log
#SBATCH --error=logs/guanlab_%A_task_%a.err

# ── 1. CONTROL PANEL ────────────────────────────────────────────────────────
INPUT_H5AD=""

# ── 2. DIRECTORY SETUP ──────────────────────────────────────────────────────
# Point BASE_DIR to the folder where your Python script lives
BASE_DIR=""

# The chunks and logs will now be safely created inside to_guanlab
CHUNK_DIR="${BASE_DIR}/chunks_"
LOG_DIR="${BASE_DIR}/logs_GuanLab"

mkdir -p "$CHUNK_DIR"
mkdir -p "$LOG_DIR"

# ── 3. ENVIRONMENT (THE FIX) ────────────────────────────────────────────────
# Pointing directly to the environment that has LightGBM installed
PYTHON_BIN=/path/to/your/python/venv

# Fail fast check: If LightGBM isn't here, kill the job immediately before wasting cluster time
$PYTHON_BIN -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')" \
    || { echo "CRITICAL ERROR: LightGBM not found in $PYTHON_BIN"; exit 1; }

# ── 4. EXECUTION ────────────────────────────────────────────────────────────
echo "Booting GuanLab Worker for Array Task: $SLURM_ARRAY_TASK_ID"
echo "Processing File: $INPUT_H5AD"

cd "$BASE_DIR"

srun $PYTHON_BIN 2_run_guanlab_worker.py \
    --input_file "$INPUT_H5AD" \
    --output_dir "$CHUNK_DIR" \
    --task_id $SLURM_ARRAY_TASK_ID \
    --total_tasks 10 \
    --n_jobs 32

echo "Task $SLURM_ARRAY_TASK_ID complete."
