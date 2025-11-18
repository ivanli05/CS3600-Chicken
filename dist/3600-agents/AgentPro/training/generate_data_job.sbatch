#!/bin/bash
#SBATCH --job-name=agentb_datagen    # Job name
#SBATCH --output=logs/datagen_%j.out # Output log
#SBATCH --error=logs/datagen_%j.err  # Error log
#SBATCH --time=03:00:00              # 3-hour time limit
#SBATCH --partition=cpu              # CPU partition (no GPU needed)
#SBATCH --cpus-per-task=32           # Use many cores for parallel generation
#SBATCH --mem=32G                    # Memory
#SBATCH --ntasks=1                   # Single task

# ============================================================================
# AgentB Training Data Generation (Parallel)
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.11

# Activate environment
# source ~/venvs/agentb/bin/activate
# or
# conda activate agentb

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to training directory
cd $SLURM_SUBMIT_DIR

echo "Working directory: $(pwd)"
echo ""

echo "Starting parallel data generation..."
echo ""

# Run data generation with all available cores
python generate_data_parallel.py \
    --config config.yaml \
    --output training_data.json \
    --num-workers $SLURM_CPUS_PER_TASK

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Show file size
if [ -f "training_data.json" ]; then
    SIZE=$(du -h training_data.json | cut -f1)
    echo "Generated file: training_data.json ($SIZE)"
    echo ""
    echo "Ready for training! Submit train_job.sbatch"
fi
